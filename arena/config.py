"""Settings from environment variables (or a local `.env` file, never committed)."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data_raw"

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "openai/gpt-oss-120b")
# Groq free tier: 8,000 tokens/minute, so the judge is paced conservatively.
LLM_MAX_QPM = int(os.getenv("LLM_MAX_QPM", "10"))
LLM_MAX_TOKENS_PER_DAY = int(os.getenv("LLM_MAX_TOKENS_PER_DAY", "400000"))

# Web app limits
MAX_CONTEXT_WORDS = int(os.getenv("MAX_CONTEXT_WORDS", "1500"))
MAX_ANSWER_WORDS = int(os.getenv("MAX_ANSWER_WORDS", "300"))
