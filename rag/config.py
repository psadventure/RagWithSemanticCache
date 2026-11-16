"""
App settings, read from the root .env (or the real environment).

Every model call goes through the LiteLLM gateway; the app only knows the gateway URL,
a virtual key and two model aliases. Which Vertex model sits behind an alias is decided
in infra/config/config.yaml.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Load the project's .env no matter where a process was started from (uvicorn, adk web, pytest).
load_dotenv(PROJECT_ROOT / ".env")


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ── LiteLLM gateway ──────────────────────────────────────────────────────────
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000").rstrip("/")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# ── Data, chunking and retrieval ─────────────────────────────────────────────
DATA_PATH = _path(os.getenv("DATA_PATH", "data"))
VECTOR_STORE_PATH = _path(os.getenv("VECTOR_STORE_PATH", "vector_store"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

# ── Semantic cache (BetterDB on Valkey) ──────────────────────────────────────
VALKEY_HOST = os.getenv("VALKEY_HOST", "localhost")
VALKEY_PORT = int(os.getenv("VALKEY_PORT", "6379"))
SEMANTIC_CACHE_NAME = os.getenv("SEMANTIC_CACHE_NAME", "ragcache")
# Cosine DISTANCE (0 = identical). A question is a hit when its distance to the closest
# cached question is <= this. Calibrated for gemini-embedding-001; see CLAUDE.md.
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.13"))
# Model name BetterDB's cost table uses to price saved calls (it knows gemini-2.5-flash).
CACHE_COST_MODEL = os.getenv("CACHE_COST_MODEL", "gemini-2.5-flash")
# BetterDB sends anonymous usage analytics unless disabled. Off unless set to true.
BETTERDB_TELEMETRY = os.getenv("BETTERDB_TELEMETRY", "false").lower() in ("1", "true", "yes", "on")

# ── Optional: AWS ElastiCache Serverless with IAM auth instead of local Valkey ──
VALKEY_USE_IAM_AUTH = os.getenv("VALKEY_USE_IAM_AUTH", "false").lower() == "true"
VALKEY_IAM_USER_ID = os.getenv("VALKEY_IAM_USER_ID", "default.iam-user")
VALKEY_SERVERLESS_CACHE_NAME = os.getenv("VALKEY_SERVERLESS_CACHE_NAME", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
