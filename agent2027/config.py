"""Every setting lives here, so the same code runs in Colab, Cloud Run and locally.

Read from environment variables. Modules read `config.X` at call time, so a
notebook can also override a value: `config.MAX_STEPS = 10`.
"""
import os
import uuid
from functools import lru_cache
from pathlib import Path


try:  # optional: load GEMINI_API_KEY etc. from a .env file (searched upward from the cwd)
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))  # real environment variables still win
except ImportError:
    pass


def _flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes")


# One line to swap the model. `gemini-flash-latest` is an alias; pin an exact
# model ID for the event (and test the fallback).
MODEL = os.getenv("AGENT_MODEL", "gemini-flash-latest")

# Auth: AI Studio key for attendees (GEMINI_API_KEY), Agent Platform for
# deployed services (GOOGLE_GENAI_USE_VERTEXAI=true + project + location).
USE_VERTEX = _flag("GOOGLE_GENAI_USE_VERTEXAI")
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))
COMPACT_AT_TOKENS = int(os.getenv("COMPACT_AT_TOKENS", "40000"))

# Tool limits are enforced by the harness, never by the model.
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
MAX_FETCH_CHARS = int(os.getenv("MAX_FETCH_CHARS", "6000"))
MAX_TOOL_CHARS = int(os.getenv("MAX_TOOL_CHARS", "12000"))
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "auto")  # or "duckduckgo", "brave", "bing", ...
SEARCH_DELAY =float(os.getenv("SEARCH_DELAY", "1.0"))

# live  = DuckDuckGo (with disk cache and backoff)
# cached = only bundled/disk-cached results (repeatable, immune to throttling)
SEARCH_MODE = os.getenv("SEARCH_MODE", "live")

# Where state lives: "memory" (one process) or "firestore" (survives restarts).
STORE = os.getenv("STORE", "memory")

# `workdir/` is ephemeral on Cloud Run. Anything that must persist goes to Firestore.
WORKDIR = Path(os.getenv("AGENT_WORKDIR", "workdir"))


def new_run_id() -> str:
    return uuid.uuid4().hex[:8]


@lru_cache(maxsize=1)
def get_client():
    from google import genai

    if USE_VERTEX:
        return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    return genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
