"""
 RAG answer generator (production-grade)
=======================================================
Responsibilities
----------------
- Intent classification    → drives retrieval depth & generation budget
- Context budgeting        → enforces MAX_CHUNKS / MAX_CHARS_PER_CHUNK limits
  before any text reaches the LLM (prevents token-overrun & cost spikes)
- Retrieval routing        → full-doc scan for summary/unit_list; top-k for all
  other intents; zero-result fallback to retrieve_all
- Prompt construction      → file-type-aware system prompt + chat history
- Groq streaming/sync      → tenacity retry on rate-limit / connection errors
- Answer cache             → MD5-keyed, thread-safe in-process LRU-style dict

Changelog
---------
[ADDED]   Per-intent context budget (MAX_CHUNKS, MAX_CHARS_PER_CHUNK) applied
          inside _build_messages() — context is truncated before LLM call.
[ADDED]   CONTEXT_BUDGET dict — each intent carries its own chunk/char limits
          so summaries get more context and quick definitions get less.
[ADDED]   "explain" intent top_k reduced to 3 (was 8) — sufficient depth,
          lower latency and cost.
[CHANGED] Updated Pinecone references → Qdrant in log messages.
[CHANGED] Added multi-file type support: PDF, DOCX, XLSX, CSV, TXT, MD,
          PPTX, JSON, HTML, XML, RTF, EPUB, email (.eml/.msg), code files.
[UNCHANGED] Groq streaming, caching, retry logic, prompt builder structure.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from config import settings
from pipeline.retriever import retrieve, retrieve_all


# ── Constants ─────────────────────────────────────────────────────────────────

# Global hard ceiling — never exceeded regardless of intent.
# Intent-specific limits in CONTEXT_BUDGET may be lower.
GLOBAL_MAX_CHUNKS: int = 20
GLOBAL_MAX_CHARS_PER_CHUNK: int = 1_200


# ---------------------------------------------------------------------------
# Per-intent context budget
# ---------------------------------------------------------------------------
# Each entry controls how much raw text is forwarded to the LLM for that
# intent.  Keeping these tight reduces token cost and avoids context bloat.
# For intents not listed here, the global constants above apply.
#
# Fields
# ------
# max_chunks          : hard cap on number of retrieved chunks sent to LLM
# max_chars_per_chunk : each chunk's content is sliced to this length
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContextBudget:
    max_chunks: int
    max_chars_per_chunk: int


CONTEXT_BUDGET: Dict[str, ContextBudget] = {
    # Quick factual answers — very tight budget
    "marks_2":       ContextBudget(max_chunks=3,  max_chars_per_chunk=600),
    "define":        ContextBudget(max_chunks=3,  max_chars_per_chunk=600),
    "example":       ContextBudget(max_chunks=3,  max_chars_per_chunk=700),

    # Standard explanation — reduced from 8 chunks to 3 per requirement
    "explain":       ContextBudget(max_chunks=3,  max_chars_per_chunk=800),

    # Medium-depth answers
    "marks_8":       ContextBudget(max_chunks=5,  max_chars_per_chunk=800),
    "list":          ContextBudget(max_chunks=6,  max_chars_per_chunk=800),
    "compare":       ContextBudget(max_chunks=6,  max_chars_per_chunk=900),
    "code_review":   ContextBudget(max_chunks=3,  max_chars_per_chunk=1_200),
    "data_analysis": ContextBudget(max_chunks=8,  max_chars_per_chunk=900),
    "default":       ContextBudget(max_chunks=5,  max_chars_per_chunk=800),

    # Deeper answers
    "marks_13":      ContextBudget(max_chunks=10, max_chars_per_chunk=1_000),

    # Full-doc intents — generous but still bounded
    "marks_16":      ContextBudget(max_chunks=15, max_chars_per_chunk=1_200),
    "summary":       ContextBudget(max_chunks=20, max_chars_per_chunk=1_200),
    "unit_list":     ContextBudget(max_chunks=20, max_chars_per_chunk=1_000),
}


def _apply_context_budget(
    chunks: List[Dict[str, Any]],
    intent_type: str,
) -> tuple[List[Dict[str, Any]], int, int]:
    """
    Slice chunks and truncate content according to the intent's budget.

    Returns
    -------
    (budgeted_chunks, max_chunks_used, max_chars_used)
    """
    budget = CONTEXT_BUDGET.get(
        intent_type,
        ContextBudget(
            max_chunks=GLOBAL_MAX_CHUNKS,
            max_chars_per_chunk=GLOBAL_MAX_CHARS_PER_CHUNK,
        ),
    )

    sliced = chunks[: budget.max_chunks]

    budgeted: List[Dict[str, Any]] = []
    for c in sliced:
        trimmed = dict(c)
        content = trimmed.get("content", "")
        if len(content) > budget.max_chars_per_chunk:
            trimmed["content"] = content[: budget.max_chars_per_chunk]
            trimmed["_truncated"] = True
        budgeted.append(trimmed)

    logger.debug(
        "Context budget applied | intent='{}' | chunks {}/{} | chars_limit={}",
        intent_type,
        len(budgeted),
        len(chunks),
        budget.max_chars_per_chunk,
    )

    return budgeted, budget.max_chunks, budget.max_chars_per_chunk


# ── Supported file types ──────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    # Documents
    ".pdf", ".docx", ".doc", ".rtf", ".odt", ".epub",
    # Spreadsheets
    ".xlsx", ".xls", ".xlsm", ".ods", ".csv", ".tsv",
    # Presentations
    ".pptx", ".ppt",
    # Text / Markup
    ".txt", ".md", ".markdown", ".html", ".htm", ".xml",
    # Data
    ".json", ".jsonl", ".yaml", ".yml",
    # Email
    ".eml", ".msg",
    # Code files (common enterprise languages)
    ".py", ".js", ".ts", ".java", ".cs", ".cpp", ".c",
    ".go", ".rb", ".php", ".sql", ".sh", ".bat", ".ps1",
    # Logs / config
    ".log", ".ini", ".toml", ".env",
})

SUPPORTED_EXTENSIONS_DISPLAY: str = ", ".join(sorted(SUPPORTED_EXTENSIONS))


def is_supported_file(filename: str) -> bool:
    """Return True if the file extension is in the supported set."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


# ── Groq client (singleton, thread-safe) ─────────────────────────────────────

_client: Optional[Groq] = None
_client_lock = threading.Lock()


def _get_client() -> Groq:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = Groq(api_key=settings.GROQ_API_KEY)
                logger.info("Groq client initialized.")
    return _client


# ── Answer cache (thread-safe) ───────────────────────────────────────────────

_answer_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()


def _cache_key(query: str, filename: str) -> str:
    raw = f"{filename}::{query.strip().lower()}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    with _cache_lock:
        return _answer_cache.get(key)


def _cache_set(key: str, value: Dict[str, Any]) -> None:
    with _cache_lock:
        _answer_cache[key] = value


# ── File type detection helpers ───────────────────────────────────────────────

def _get_file_type_label(filename: str) -> str:
    """Return a human-readable file type label for prompt context."""
    ext = Path(filename).suffix.lower()
    labels: Dict[str, str] = {
        ".pdf":  "PDF document",
        ".docx": "Word document",  ".doc": "Word document",
        ".xlsx": "Excel spreadsheet", ".xls": "Excel spreadsheet",
        ".xlsm": "Excel spreadsheet", ".ods": "spreadsheet",
        ".csv":  "CSV data file",  ".tsv": "TSV data file",
        ".pptx": "PowerPoint presentation", ".ppt": "PowerPoint presentation",
        ".txt":  "text file",      ".md": "Markdown document",
        ".html": "HTML document",  ".htm": "HTML document",
        ".xml":  "XML file",
        ".json": "JSON data file", ".jsonl": "JSONL data file",
        ".yaml": "YAML config file", ".yml": "YAML config file",
        ".eml":  "email file",     ".msg": "email file",
        ".py":   "Python source file", ".js": "JavaScript source file",
        ".ts":   "TypeScript source file", ".java": "Java source file",
        ".cs":   "C# source file", ".cpp": "C++ source file",
        ".sql":  "SQL file",       ".log": "log file",
        ".rtf":  "RTF document",   ".epub": "eBook",
    }
    return labels.get(ext, "document")


def _get_file_type_instructions(filename: str) -> str:
    """Return file-type-specific system instructions for the LLM."""
    ext = Path(filename).suffix.lower()

    if ext in {".xlsx", ".xls", ".xlsm", ".ods", ".csv", ".tsv"}:
        return (
            "This is a spreadsheet or data file. "
            "When answering, reference specific column names, row values, "
            "or sheet names when relevant. For numerical data, provide "
            "accurate figures and avoid rounding unless asked."
        )
    if ext in {".pptx", ".ppt"}:
        return (
            "This is a presentation file. "
            "Reference slide numbers when possible. "
            "Summarize key points from each slide concisely."
        )
    if ext in {".py", ".js", ".ts", ".java", ".cs", ".cpp",
               ".c", ".go", ".rb", ".php", ".sql", ".sh", ".bat", ".ps1"}:
        return (
            "This is a source code file. "
            "When explaining code, describe what functions/classes do, "
            "highlight important logic, and use code formatting in your answer."
        )
    if ext in {".json", ".jsonl", ".yaml", ".yml", ".xml"}:
        return (
            "This is a structured data/config file. "
            "Reference specific keys, fields, or values when answering. "
            "Explain the structure clearly."
        )
    if ext in {".eml", ".msg"}:
        return (
            "This is an email file. "
            "Pay attention to sender, recipient, subject, date, and body content. "
            "Summarize or answer about the email's contents accurately."
        )
    if ext in {".log"}:
        return (
            "This is a log file. "
            "Focus on errors, warnings, timestamps, and patterns. "
            "Highlight any critical issues or anomalies."
        )
    if ext in {".html", ".htm"}:
        return (
            "This is an HTML document. "
            "Focus on the visible text content and structure, "
            "ignoring raw HTML tags unless specifically asked about them."
        )
    return "Answer based strictly on the document content provided."


# ── Intent detection ──────────────────────────────────────────────────────────

# Intents that require the full document rather than top-k retrieval.
_FULL_DOC_INTENTS: frozenset[str] = frozenset({"summary", "unit_list"})


def _detect_intent(query: str) -> Dict[str, Any]:
    """
    Classify query intent and return retrieval + generation parameters.

    Design rules
    ------------
    - Evaluated top-to-bottom; first match wins.
    - Always run on the ORIGINAL (uncleaned) query.
    - top_k for "explain" is 3 (reduced per project requirement).
    - All intents declare max_tokens, top_k, and instruction.
    """
    q = query.lower().strip()

    # ── Mark-based academic intents ──────────────────────────────────────────
    if re.search(r"16\+?\s*marks?|20\s*marks?", q):
        return {
            "intent_type": "marks_16",
            "max_tokens": 2048,
            "top_k": 15,
            "instruction": (
                "Give a comprehensive, well-structured answer with headings, "
                "sub-points, and examples. Cover every aspect thoroughly as "
                "required for a 16-mark university answer."
            ),
        }

    if re.search(r"1[0-3]\s*marks?", q):
        return {
            "intent_type": "marks_13",
            "max_tokens": 1024,
            "top_k": 10,
            "instruction": (
                "Give a detailed explanation with an example. "
                "Suitable depth for a 10-13 mark answer."
            ),
        }

    if re.search(r"[5-8]\s*marks?", q):
        return {
            "intent_type": "marks_8",
            "max_tokens": 600,
            "top_k": 6,
            "instruction": (
                "Give a moderate explanation with a short example. "
                "Suitable for a 5-8 mark answer."
            ),
        }

    if re.search(r"2\s*marks?", q):
        return {
            "intent_type": "marks_2",
            "max_tokens": 300,
            "top_k": 3,
            "instruction": "Give a brief 3-5 line answer. Be concise.",
        }

    # ── Data / spreadsheet intents ───────────────────────────────────────────
    if re.search(
        r"\b(calculate|compute|total|sum|average|mean|count|max|min|"
        r"highest|lowest|filter|sort|group by|pivot|chart|graph|"
        r"how many rows?|how many columns?|what columns?|column names?)\b",
        q,
    ):
        return {
            "intent_type": "data_analysis",
            "max_tokens": 800,
            "top_k": 10,
            "instruction": (
                "Analyze the data carefully. Reference specific column names, "
                "values, and provide accurate numerical answers. "
                "Show your reasoning step by step."
            ),
        }

    # ── Code intents ─────────────────────────────────────────────────────────
    if re.search(
        r"\b(function|class|method|variable|bug|error|fix|refactor|"
        r"what does this code|explain the code|how does this work|"
        r"import|dependency|module|test|debug)\b",
        q,
    ):
        return {
            "intent_type": "code_review",
            "max_tokens": 1200,
            "top_k": 3,
            "instruction": (
                "Analyze the code carefully. Explain what it does, "
                "identify any issues if asked, and use code formatting "
                "in your response where appropriate."
            ),
        }

    # ── Summary / overview intents ───────────────────────────────────────────
    if re.search(
        r"\b(summarize|summarise|summary|overview|brief|outline|gist|recap|"
        r"entire|whole|full|complete|all units|all topics|all chapters|"
        r"what is (this|the) (document|pdf|book|notes?|file|spreadsheet|presentation) about|"
        r"what does (this|the) (document|pdf|book|notes?|file) (cover|contain|discuss)|"
        r"give me an (overview|summary)|tell me about (this|the) (document|pdf|file))\b",
        q,
    ):
        return {
            "intent_type": "summary",
            "max_tokens": 2000,
            "top_k": 20,
            "instruction": (
                "You have been given the FULL content of the document in chunks. "
                "Your task is to produce a complete, well-structured summary. "
                "Structure your response as follows:\n"
                "1. **Document Overview** — what the document is about in 2-3 sentences.\n"
                "2. **Key Topics Covered** — list the main subjects/units/chapters with a brief description.\n"
                "3. **Important Concepts** — highlight 5-10 key concepts, definitions, or ideas.\n"
                "4. **Conclusion** — a closing summary sentence.\n\n"
                "Use ALL the context provided. Do NOT say information is missing — "
                "synthesize everything that is present. Be thorough and detailed."
            ),
        }

    # ── Unit / topic listing intent ──────────────────────────────────────────
    if re.search(
        r"\b(list|what are|show|give).{0,25}\b(units?|topics?|chapters?|sections?|modules?|contents?|syllabus)\b",
        q,
    ):
        return {
            "intent_type": "unit_list",
            "max_tokens": 1000,
            "top_k": 20,
            "instruction": (
                "List ONLY the units, topics, chapters, or sections that are "
                "explicitly present in the provided context. "
                "Format as a numbered list with the exact names as they appear "
                "in the document. Do NOT invent anything not in the context."
            ),
        }

    # ── General list intent ──────────────────────────────────────────────────
    if re.search(r"\b(list|enumerate|types of|kinds of|examples of)\b", q):
        return {
            "intent_type": "list",
            "max_tokens": 700,
            "top_k": 8,
            "instruction": (
                "List ALL relevant items found in the context with brief "
                "explanations. Do not add items not present in the context."
            ),
        }

    # ── Explanation intents ──────────────────────────────────────────────────
    # top_k=3 per project requirement (was 8)
    if re.search(r"\b(explain|describe|discuss|elaborate)\b", q):
        return {
            "intent_type": "explain",
            "max_tokens": 1024,
            "top_k": 3,
            "instruction": (
                "Give a detailed, well-structured explanation with examples "
                "where appropriate."
            ),
        }

    # ── Comparison intents ───────────────────────────────────────────────────
    if re.search(r"\b(compare|differentiate|difference|vs\.?|versus)\b", q):
        return {
            "intent_type": "compare",
            "max_tokens": 900,
            "top_k": 8,
            "instruction": (
                "Compare clearly using a structured table or labeled points. "
                "Cover both similarities and differences."
            ),
        }

    # ── Definition intents ───────────────────────────────────────────────────
    if re.search(r"\b(define|what is|what are|meaning of)\b", q):
        return {
            "intent_type": "define",
            "max_tokens": 500,
            "top_k": 5,
            "instruction": "Give a clear definition with a short example.",
        }

    # ── Example intents ──────────────────────────────────────────────────────
    if re.search(r"\b(example|illustrate|show)\b", q):
        return {
            "intent_type": "example",
            "max_tokens": 600,
            "top_k": 5,
            "instruction": "Explain with clear, concrete examples.",
        }

    # ── Default ──────────────────────────────────────────────────────────────
    return {
        "intent_type": "default",
        "max_tokens": 600,
        "top_k": 6,
        "instruction": "Give a clear, accurate, and helpful answer.",
    }


# ── Retrieval router ──────────────────────────────────────────────────────────

def _retrieve_chunks(
    intent: Dict[str, Any],
    retrieval_query: str,
    filename: str,
) -> List[Dict[str, Any]]:
    """
    Route retrieval based on intent type.

    - Full-doc intents (summary, unit_list) → retrieve_all()
    - All others                            → retrieve() with top-k
    - Zero-result fallback                  → retrieve_all() (safety net)
    """
    intent_type = intent["intent_type"]

    if intent_type in _FULL_DOC_INTENTS:
        logger.info(
            "Intent '{}' → retrieve_all() | file='{}'",
            intent_type, filename,
        )
        chunks = retrieve_all(filename, max_tokens=6000)

        if not chunks:
            logger.warning(
                "retrieve_all() returned 0 chunks | intent='{}' file='{}'. "
                "Possible causes: file not indexed, collection mismatch, or Qdrant empty.",
                intent_type, filename,
            )
        else:
            logger.info("retrieve_all() → {} chunks.", len(chunks))

        return chunks

    # Standard hybrid retrieval
    chunks = retrieve(retrieval_query, filename, k=intent["top_k"])
    logger.info(
        "retrieve() → {} chunks | intent='{}' | top_k={} | query='{}'",
        len(chunks), intent_type, intent["top_k"], retrieval_query[:60],
    )

    # Fallback: zero results → try retrieve_all as a safety net
    if not chunks:
        logger.warning(
            "Top-k retrieval returned 0 chunks for query='{}'. "
            "Falling back to retrieve_all().",
            retrieval_query[:60],
        )
        chunks = retrieve_all(filename, max_tokens=4000)
        if chunks:
            logger.info("Fallback retrieve_all() → {} chunks.", len(chunks))

    return chunks


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_messages(
    query: str,
    context_chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
    instruction: str,
    intent_type: str,
    filename: str = "",
) -> List[Dict[str, str]]:
    """
    Construct the message list sent to the Groq LLM.

    Context budgeting is applied here, immediately before prompt assembly,
    so the limit is enforced regardless of how many chunks were retrieved.
    """
    # ── Apply per-intent context budget ──────────────────────────────────────
    budgeted_chunks, max_chunks_used, max_chars_used = _apply_context_budget(
        context_chunks, intent_type
    )

    context_text = "\n\n---\n\n".join(
        f"[Page/Section {c.get('page', '?')}] {c['content']}"
        for c in budgeted_chunks
    )

    file_type_label = _get_file_type_label(filename) if filename else "document"
    file_type_instructions = (
        _get_file_type_instructions(filename) if filename else ""
    )

    truncation_notice = (
        f"\n\nNOTE: Context was limited to {len(budgeted_chunks)} chunks "
        f"(max {max_chars_used} chars each) for this intent type."
        if len(context_chunks) > max_chunks_used
        else ""
    )

    system_prompt = (
        f"You are a precise and reliable assistant specialized in analyzing "
        f"business documents and files. Your job is to answer questions "
        f"strictly based on the {file_type_label} content provided below."
        f"\n\nFILE TYPE: {file_type_label.upper()}"
        f"\n{file_type_instructions}"
        f"\n\nRULES YOU MUST FOLLOW:"
        f"\n1. Answer ONLY from the context provided. Do not use outside knowledge."
        f"\n2. If asked to summarize — synthesize EVERYTHING available in the context "
        f"into a structured, detailed summary. You have the full document. Use it all."
        f"\n3. If asked to list units/topics — list ONLY what is explicitly present "
        f"in the context. Never invent or guess."
        f"\n4. If the context genuinely does not contain the answer, say exactly: "
        f"\"The document does not contain information about this topic.\""
        f"\n5. Never say information is missing if it IS present — read every chunk "
        f"carefully before responding."
        f"\n6. Be factually precise. Do not add, assume, or extrapolate beyond "
        f"what the context states."
        f"\n7. Format clearly: use numbered lists, headings, bold text, tables, "
        f"or paragraphs as appropriate for the file type."
        f"\n8. For spreadsheet/data files — always reference column names and specific values."
        f"\n9. For code files — use code blocks in your response and explain logic clearly."
        f"\n10. For presentations — reference slide numbers when possible."
        f"\n\nCONTEXT FROM {file_type_label.upper()} "
        f"({len(budgeted_chunks)}/{len(context_chunks)} chunks used):"
        f"{truncation_notice}"
        f"\n\n{context_text}"
        f"\n\nRESPONSE INSTRUCTION: {instruction}"
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Include last 6 turns of chat history for follow-up support
    for msg in chat_history[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})
    return messages


# ── Groq API calls with tenacity retry ───────────────────────────────────────

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_groq_stream(messages: List[Dict[str, str]], max_tokens: int):
    """Streaming Groq call with automatic retry on transient errors."""
    return _get_client().chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
        stream=True,
    )


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_groq_sync(messages: List[Dict[str, str]], max_tokens: int):
    """Non-streaming Groq call with automatic retry on transient errors."""
    return _get_client().chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
        stream=False,
    )


# ── Public API — Streaming ────────────────────────────────────────────────────

def answer_query_stream(
    query: str,
    chat_history: List[Dict[str, str]],
    filename: str,
) -> Generator[str, None, None]:
    """
    Stream the answer token-by-token.

    Pipeline
    --------
    1. Validate filename / extension
    2. Check answer cache
    3. Detect intent (on original query)
    4. Clean query for retrieval
    5. Retrieve chunks
    6. Apply context budget (inside _build_messages)
    7. Stream from Groq
    8. Cache result
    """
    if not filename:
        yield (
            "Please upload a file first before asking questions. "
            f"Supported formats: {SUPPORTED_EXTENSIONS_DISPLAY}"
        )
        return

    if not is_supported_file(filename):
        ext = Path(filename).suffix.lower()
        yield (
            f"Sorry, '{ext}' files are not supported. "
            f"Supported formats: {SUPPORTED_EXTENSIONS_DISPLAY}"
        )
        return

    # ── Cache check ───────────────────────────────────────────────────────────
    key = _cache_key(query, filename)
    cached = _cache_get(key)
    if cached:
        logger.info("Cache hit | query='{}'", query[:60])
        answer = cached["answer"]
        for i in range(0, len(answer), 8):
            yield answer[i : i + 8]
        return

    # ── Intent detection on ORIGINAL query ───────────────────────────────────
    intent = _detect_intent(query)
    logger.info(
        "Intent='{}' | file_type='{}' | top_k={} | query='{}'",
        intent["intent_type"],
        _get_file_type_label(filename),
        intent["top_k"],
        query[:60],
    )

    # ── Clean query for retrieval only ────────────────────────────────────────
    retrieval_query = _clean_query(query)

    # ── Retrieve chunks ───────────────────────────────────────────────────────
    chunks = _retrieve_chunks(intent, retrieval_query, filename)

    if not chunks:
        logger.error(
            "Zero chunks returned | query='{}' file='{}'. "
            "Check if file was indexed in Qdrant.",
            query[:60], filename,
        )
        yield (
            "I could not retrieve content from this file. "
            "This usually means the file was not indexed yet. "
            "Please try re-uploading the file and ask again."
        )
        return

    # ── Build messages (budget applied inside) ────────────────────────────────
    messages = _build_messages(
        query,
        chunks,
        chat_history,
        intent["instruction"],
        intent["intent_type"],
        filename,
    )

    # ── Stream from Groq ──────────────────────────────────────────────────────
    full_answer = ""
    try:
        stream = _call_groq_stream(messages, intent["max_tokens"])
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_answer += token
            yield token

    except RateLimitError:
        logger.error("Groq rate limit hit after retries.")
        yield "\n\n[Rate limit reached. Please wait a moment and try again.]"
        return

    except APIConnectionError as exc:
        logger.error("Groq connection error: {}", exc)
        yield "\n\n[Connection error. Please check your internet and try again.]"
        return

    except APIStatusError as exc:
        logger.error("Groq API error {}: {}", exc.status_code, exc.message)
        yield f"\n\n[API error {exc.status_code}. Please try again.]"
        return

    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected Groq error: {}", exc)
        yield f"\n\n[Unexpected error: {exc}]"
        return

    # ── Cache the completed answer ────────────────────────────────────────────
    if full_answer.strip():
        _cache_set(key, {"answer": full_answer, "sources": chunks})
        logger.info(
            "Answer cached | intent='{}' | file_type='{}' | chunks_used={}/{} | tokens~={}",
            intent["intent_type"],
            _get_file_type_label(filename),
            min(len(chunks), CONTEXT_BUDGET.get(
                intent["intent_type"],
                ContextBudget(GLOBAL_MAX_CHUNKS, GLOBAL_MAX_CHARS_PER_CHUNK),
            ).max_chunks),
            len(chunks),
            len(full_answer.split()),
        )


# ── Public API — Non-streaming ────────────────────────────────────────────────

def answer_query(
    query: str,
    chat_history: List[Dict[str, str]],
    filename: str,
) -> Dict[str, Any]:
    """
    Non-streaming version — returns ``{"answer": str, "sources": list}``.
    Reads from cache if available.
    """
    if not filename:
        return {
            "answer": (
                "Please upload a file first. "
                f"Supported formats: {SUPPORTED_EXTENSIONS_DISPLAY}"
            ),
            "sources": [],
        }

    if not is_supported_file(filename):
        ext = Path(filename).suffix.lower()
        return {
            "answer": (
                f"Sorry, '{ext}' files are not supported. "
                f"Supported formats: {SUPPORTED_EXTENSIONS_DISPLAY}"
            ),
            "sources": [],
        }

    key = _cache_key(query, filename)
    cached = _cache_get(key)
    if cached:
        return cached

    intent = _detect_intent(query)
    retrieval_query = _clean_query(query)
    chunks = _retrieve_chunks(intent, retrieval_query, filename)

    if not chunks:
        return {
            "answer": (
                "I could not retrieve content from this file. "
                "Please try re-uploading the file and ask again."
            ),
            "sources": [],
        }

    messages = _build_messages(
        query,
        chunks,
        chat_history,
        intent["instruction"],
        intent["intent_type"],
        filename,
    )

    try:
        response = _call_groq_sync(messages, intent["max_tokens"])
        answer: str = response.choices[0].message.content or ""
    except RateLimitError:
        answer = "Rate limit reached. Please wait a moment and try again."
    except APIConnectionError:
        answer = "Connection error. Please check your internet and try again."
    except Exception as exc:  # noqa: BLE001
        logger.error("Groq error: {}", exc)
        answer = f"Error generating answer: {exc}"

    result: Dict[str, Any] = {"answer": answer, "sources": chunks}
    _cache_set(key, result)
    return result


# ── Query cleaner ─────────────────────────────────────────────────────────────

def _clean_query(query: str) -> str:
    """
    Strip mark-count references and filler prefixes before vector retrieval.

    IMPORTANT: Always call AFTER intent detection — never before.
    """
    cleaned = re.sub(
        r"\bfor\s+\d+\+?\s*marks?\b", "", query, flags=re.IGNORECASE
    ).strip()
    cleaned = re.sub(
        r"^(can you|please|tell me|i want to know|give me|show me)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or query


# ── Cache management ──────────────────────────────────────────────────────────

def clear_cache(filename: str) -> None:
    """Evict all cached answers. Call when a file is deleted or switched."""
    with _cache_lock:
        _answer_cache.clear()
    logger.info("Answer cache cleared | triggered by file switch: '{}'", filename)


def get_cache_stats() -> Dict[str, int]:
    """Return cache statistics for observability / debugging."""
    with _cache_lock:
        return {"cached_queries": len(_answer_cache)}