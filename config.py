"""
MedQueryAI - Central Configuration
All tunable parameters in one place for easy experimentation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# ──────────────────────────────────────────────
# API Keys (supports multiple LLM providers)
# ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ──────────────────────────────────────────────
# Chunking Parameters
# ──────────────────────────────────────────────
CHUNK_SIZE = 512            # tokens per chunk
CHUNK_OVERLAP = 64          # token overlap between chunks
MIN_CHUNK_SIZE = 50         # minimum chunk size (skip tiny fragments)

# Medical section headers to detect for section-aware splitting
MEDICAL_SECTION_HEADERS = [
    "chief complaint",
    "history of present illness",
    "past medical history",
    "medications",
    "allergies",
    "family history",
    "social history",
    "review of systems",
    "physical examination",
    "vital signs",
    "laboratory results",
    "lab results",
    "imaging",
    "assessment",
    "diagnosis",
    "plan",
    "treatment plan",
    "discharge summary",
    "discharge instructions",
    "follow-up",
    "procedures",
    "operative note",
    "clinical findings",
    "impression",
    "recommendations",
    "prognosis",
]

# ──────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384   # dimension for all-MiniLM-L6-v2

# ──────────────────────────────────────────────
# ChromaDB
# ──────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "medquery_docs"

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K_RESULTS = 5           # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.15  # minimum cosine similarity score to include

# ──────────────────────────────────────────────
# LLM Configuration
# ──────────────────────────────────────────────
# Supported providers: "gemini" (free), "groq" (free), "claude"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Model names per provider
LLM_MODELS = {
    "gemini": "gemini-2.0-flash",        # FREE — 1500 req/day
    "groq": "llama-3.3-70b-versatile",   # FREE tier available
    "claude": "claude-sonnet-4-20250514", # Paid
}

LLM_TEMPERATURE = 0.1       # low temp for factual medical answers
LLM_MAX_TOKENS = 1024       # max response tokens

# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
APP_TITLE = "MedQueryAI"
APP_SUBTITLE = "Clinical Document RAG Engine"
APP_ICON = "🏥"
MAX_UPLOAD_SIZE_MB = 50

