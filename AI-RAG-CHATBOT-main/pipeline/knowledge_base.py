import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
from typing import List, Dict, Any
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
)
from config import settings
from loaders.pdf_loader import load_pdf
import torch

# ── CHANGELOG (Session 10) ────────────────────────────────────────────────────
# [CHANGED] Embedding model: BGE-small-en-v1.5 → all-MiniLM-L6-v2
#           Reason: 3s load vs 9s, 80MB vs 130MB, frees VRAM for Wav2Lip + face model
# [CHANGED] MINILM_DIM = 384 (same dimension, Qdrant collection compatible)
# [CHANGED] Removed BGE passage prefix — MiniLM does not need instruction prefix
# [UNCHANGED] Qdrant storage path, chunking logic, ingest flow, singleton pattern
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# all-MiniLM-L6-v2 embedding dimension
MINILM_DIM = 384

# ── Singleton clients ─────────────────────────────────────────────────────────
_qdrant_client: QdrantClient | None = None
_embed_model:   SentenceTransformer | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton QdrantClient (local disk storage)."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path="./data/qdrant_storage")
        logger.info("Qdrant client connected (local storage).")
    return _qdrant_client


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("Loading all-MiniLM-L6-v2 model...")
        _embed_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=DEVICE
        )
        logger.info("Embedding model loaded.")
    return _embed_model


# ── Collection name helpers ───────────────────────────────────────────────────

def _collection_name(filename: str) -> str:
    """Derive a valid Qdrant collection name from a filename."""
    name = Path(filename).stem
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_-')
    name = name[:50]
    name = name.strip('_-')
    if len(name) < 3:
        name = name + "col"
    return name.lower()


def get_collection(filename: str = None) -> str:
    return _collection_name(filename) if filename else settings.COLLECTION_NAME


def _ensure_collection(client: QdrantClient, collection: str) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=MINILM_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '{}'.", collection)


def is_indexed(filename: str) -> bool:
    """Return True if the file has already been ingested into Qdrant."""
    try:
        client     = get_qdrant_client()
        collection = get_collection(filename)
        existing   = {c.name for c in client.get_collections().collections}
        if collection not in existing:
            return False
        info = client.get_collection(collection)
        return (info.points_count or 0) > 0
    except Exception:
        return False


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed(texts: List[str]) -> List[List[float]]:
    model = get_embed_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
        device=DEVICE,
        convert_to_numpy=True,
)
    return embeddings.tolist()

# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_document(file_path: str) -> int:
    source = Path(file_path).name

    pages       = load_pdf(file_path)
    total_text  = " ".join([p["text"] for p in pages])
    total_words = len(total_text.split())

    if total_words < 500:
        chunk_size = 60
        overlap    = 10
    elif total_words < 2000:
        chunk_size = 100
        overlap    = 15
    else:
        chunk_size = settings.CHUNK_SIZE
        overlap    = settings.CHUNK_OVERLAP

    logger.info(
        "Document '{}' has {} words - using chunk_size={}",
        source, total_words, chunk_size,
    )

    # ── Skip if already indexed ───────────────────────────────────────────────
    if is_indexed(source):
        client     = get_qdrant_client()
        collection = get_collection(source)
        info       = client.get_collection(collection)
        count      = info.points_count or 0
        logger.info("'{}' already indexed with {} vectors — skipping.", source, count)
        return count

    # ── Build chunks ──────────────────────────────────────────────────────────
    chunks, metadatas, ids = [], [], []
    chunk_idx = 0

    for page in pages:
        page_chunks = _chunk_text(page["text"], chunk_size, overlap)
        for chunk in page_chunks:
            if len(chunk.strip()) < 10:
                continue
            chunks.append(chunk)
            metadatas.append({"source": source, "page": page["page"]})
            ids.append(chunk_idx)
            chunk_idx += 1

    if not chunks:
        logger.warning("No chunks extracted from {}", file_path)
        return 0

    # ── Embed ─────────────────────────────────────────────────────────────────
    embeddings = _embed(chunks)

    # ── Upsert to Qdrant ──────────────────────────────────────────────────────
    client     = get_qdrant_client()
    collection = get_collection(source)
    _ensure_collection(client, collection)

    points = [
        PointStruct(
            id=ids[i],
            vector=embeddings[i],
            payload={
                "text":   chunks[i],
                "source": metadatas[i]["source"],
                "page":   metadatas[i]["page"],
            },
        )
        for i in range(len(chunks))
    ]

    BATCH = 100
    for i in range(0, len(points), BATCH):
        batch = points[i : i + BATCH]
        client.upsert(collection_name=collection, points=batch)
        logger.debug(
            "Upserted batch {}/{}",
            i // BATCH + 1,
            (len(points) - 1) // BATCH + 1,
        )

    logger.info(
        "Ingested {} chunks from '{}' into Qdrant collection '{}'",
        len(chunks), source, collection,
    )
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        count = ingest_document(sys.argv[1])
        print(f"Ingested {count} chunks")