import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any, Optional
from loguru import logger
from rank_bm25 import BM25Okapi
from config import settings

# ── CHANGELOG (Session 10) ────────────────────────────────────────────────────
# [CHANGED] Embedding prefix removed — MiniLM-L6-v2 needs no instruction prefix
# [UNCHANGED] query_points() fix, shared singleton, BM25 hybrid, all retrieval logic
# ─────────────────────────────────────────────────────────────────────────────

from pipeline.knowledge_base import (
    get_qdrant_client as get_client,
    get_collection    as get_collection_name,
    get_embed_model,        # shared singleton — loads ONCE for entire app
)


# ---------------------------------------------------------------------------
# Embedding  (reuses singleton from knowledge_base — zero extra load time)
# ---------------------------------------------------------------------------
def _embed_query(query: str) -> List[float]:
    """
    Embed a query string using the shared MiniLM-L6-v2 model.
    MiniLM works best with plain text — no instruction prefix needed.
    """
    model  = get_embed_model()
    vector = model.encode(query, normalize_embeddings=True)
    return vector.tolist()


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------
def _bm25_scores(query: str, docs: List[str]) -> List[float]:
    tokenized_docs  = [d.lower().split() for d in docs]
    tokenized_query = query.lower().split()
    bm25 = BM25Okapi(tokenized_docs)
    return bm25.get_scores(tokenized_query).tolist()


def _normalize(scores: List[float]) -> List[float]:
    min_s = min(scores)
    max_s = max(scores)
    span  = max_s - min_s
    if span == 0:
        return [0.0] * len(scores)
    return [(s - min_s) / span for s in scores]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Standard hybrid retrieval (top-k)
# ---------------------------------------------------------------------------
def retrieve(query: str, filename: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
    if k is None:
        k = int(settings.TOP_K)

    client          = get_client()
    collection_name = get_collection_name(filename)

    logger.info("retrieve: query='{}' | collection='{}' | k={}", query[:60], collection_name, k)

    try:
        info      = client.get_collection(collection_name)
        vec_count = info.points_count or 0
        logger.info("retrieve: collection '{}' has {} points.", collection_name, vec_count)
    except Exception as e:
        logger.error("retrieve: collection '{}' not found or Qdrant error: {}", collection_name, e)
        return []

    if vec_count == 0:
        logger.warning("retrieve: collection '{}' is EMPTY.", collection_name)
        return []

    pool_size = min(k * 3, 30, vec_count)

    try:
        query_vector = _embed_query(query)
    except Exception as e:
        logger.error("retrieve: embedding failed: {}", e)
        return []

    # ── query_points() — Qdrant v1.7+ API (with search() fallback) ───────────
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=pool_size,
            with_payload=True,
        )
        results = response.points
    except AttributeError:
        try:
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=pool_size,
                with_payload=True,
            )
        except Exception as e:
            logger.error("retrieve: Qdrant search fallback also failed: {}", e)
            return []
    except Exception as e:
        logger.error("retrieve: Qdrant query_points failed: {}", e)
        return []

    if not results:
        logger.warning("retrieve: Qdrant returned 0 matches for collection='{}'.", collection_name)
        return []

    docs        = [r.payload.get("text", "") for r in results]
    payloads    = [r.payload                 for r in results]
    cosine_sims = [r.score                   for r in results]

    bm25_raw    = _bm25_scores(query, docs)
    cosine_norm = _normalize(cosine_sims)
    bm25_norm   = _normalize(bm25_raw)

    VECTOR_WEIGHT = 0.6
    BM25_WEIGHT   = 0.4

    fused = [
        (VECTOR_WEIGHT * c) + (BM25_WEIGHT * b)
        for c, b in zip(cosine_norm, bm25_norm)
    ]

    ranked = sorted(
        zip(fused, docs, payloads),
        key=lambda x: x[0],
        reverse=True,
    )[:k]

    chunks = [
        {
            "content": doc,
            "source":  meta.get("source", filename),
            "page":    meta.get("page", 0),
            "score":   round(score, 4),
        }
        for score, doc, meta in ranked
        if doc.strip()
    ]

    logger.info("retrieve: returned {} chunks | pool={} | collection='{}'", len(chunks), pool_size, collection_name)
    return chunks


# ---------------------------------------------------------------------------
# Full document retrieval (for summary / unit_list intents)
# ---------------------------------------------------------------------------
def retrieve_all(filename: str, max_tokens: int = 6000) -> List[Dict[str, Any]]:
    client          = get_client()
    collection_name = get_collection_name(filename)

    logger.info("retrieve_all: file='{}' | collection='{}' | max_tokens={}", filename, collection_name, max_tokens)

    try:
        info      = client.get_collection(collection_name)
        vec_count = info.points_count or 0
        logger.info("retrieve_all: collection '{}' has {} points.", collection_name, vec_count)
    except Exception as e:
        logger.error("retrieve_all: collection '{}' not found: {}", collection_name, e)
        return []

    if vec_count == 0:
        logger.error("retrieve_all: collection '{}' is EMPTY.", collection_name)
        return []

    docs_meta: List[tuple] = []
    SCROLL_BATCH = 100
    offset       = None

    try:
        while True:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=SCROLL_BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                text    = payload.get("text", "").strip()
                if text:
                    docs_meta.append((text, payload))
            if next_offset is None:
                break
            offset = next_offset
    except Exception as e:
        logger.error("retrieve_all: Qdrant scroll failed: {}", e)
        if not docs_meta:
            return []

    if not docs_meta:
        logger.error("retrieve_all: scrolled all points but payload text is empty for collection='{}'.", collection_name)
        return []

    logger.info("retrieve_all: scrolled {} chunks from Qdrant for collection='{}'.", len(docs_meta), collection_name)

    docs_meta.sort(key=lambda x: x[1].get("page", 0))

    selected:    List[Dict[str, Any]] = []
    token_count: int                  = 0

    for doc, meta in docs_meta:
        chunk_tokens = _estimate_tokens(doc)
        if token_count + chunk_tokens > max_tokens:
            logger.info("retrieve_all: token budget ({}) reached at chunk {}. Stopping.", max_tokens, len(selected))
            break
        selected.append({
            "content": doc,
            "source":  meta.get("source", filename),
            "page":    meta.get("page", 0),
            "score":   1.0,
        })
        token_count += chunk_tokens

    logger.info(
        "retrieve_all: DONE | collection='{}' | total_in_qdrant={} | fetched={} | selected={} | est_tokens={}",
        collection_name, vec_count, len(docs_meta), len(selected), token_count,
    )
    return selected