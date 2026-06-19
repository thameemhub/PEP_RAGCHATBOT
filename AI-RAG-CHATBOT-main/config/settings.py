from pydantic_settings import BaseSettings

# ── CHANGELOG (Session 11) ────────────────────────────────────────────────────
# [ADDED] DID_API_KEY — D-ID avatar API key for lip sync + facial animation
# [CHANGED] Added type annotations to QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
# [REMOVED] JINA_API_KEY         — BGE-small-v1.5 is local, no API key needed
# [REMOVED] PINECONE_API_KEY     — migrated to Qdrant
# [REMOVED] PINECONE_INDEX_NAME  — migrated to Qdrant
# [FIXED]   COLLECTION_NAME defined twice — deduplicated, kept annotated version
# [UNCHANGED] Everything else
# ──────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    # LLM + STT — Groq, one key
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # D-ID Avatar — lip sync + facial expressions
    DID_API_KEY: str = ""

    # Vector DB — Qdrant (local by default, no API key needed for local mode)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    COLLECTION_NAME: str = "knowledge_base"

    # Chunking
    TOP_K: int = 3
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 30
    # STT
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()