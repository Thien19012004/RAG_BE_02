"""
model_factory.py — Configurable LLM Factory for RAG Service

Reads LLM configuration from the system_config table and returns
the appropriate LangChain chat model instance.

Supports 3 providers: OpenAI, Groq, Gemini.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel

# Provider imports (lazy-loaded to avoid import errors if not installed)
_PROVIDER_CACHE: Dict[str, Any] = {}

# In-memory config cache with TTL
_CONFIG_CACHE: Dict[str, Any] = {}
_CACHE_TTL = 300  # 5 minutes
_CACHE_TIMESTAMPS: Dict[str, float] = {}

# Default configs (fallback when DB is unavailable)
DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "llm.generation": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.2},
    "llm.hyde": {"provider": "groq", "model": "llama-3.3-70b-versatile", "temperature": 0.7},
    "llm.condense": {"provider": "groq", "model": "llama-3.3-70b-versatile", "temperature": 0.3},
    "llm.classification": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.1},
    "llm.summarization": {"provider": "groq", "model": "llama-3.3-70b-versatile", "temperature": 0.3},
    "llm.vision": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.2},
}


def _get_db_connection():
    """Get a psycopg2 connection from env vars."""
    try:
        import psycopg2
        return psycopg2.connect(os.getenv("DATABASE_URL", ""))
    except Exception:
        return None


def get_system_config(key: str) -> Dict[str, Any]:
    """
    Load a config value from DB with caching.
    Falls back to DEFAULT_CONFIGS if DB unavailable.
    """
    now = time.time()

    # Check cache
    if key in _CONFIG_CACHE:
        if now - _CACHE_TIMESTAMPS.get(key, 0) < _CACHE_TTL:
            return _CONFIG_CACHE[key]

    # Try DB
    try:
        conn = _get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM system_config WHERE key = %s",
                (key,),
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row:
                import json
                value = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                _CONFIG_CACHE[key] = value
                _CACHE_TIMESTAMPS[key] = now
                return value
    except Exception as e:
        print(f"[model_factory] DB error reading {key}: {e}")

    # Fallback to defaults
    default = DEFAULT_CONFIGS.get(key, DEFAULT_CONFIGS["llm.generation"])
    _CONFIG_CACHE[key] = default
    _CACHE_TIMESTAMPS[key] = now
    return default


def get_llm(purpose: str) -> BaseChatModel:
    """
    Get an LLM instance for the given purpose.

    Args:
        purpose: One of "generation", "hyde", "condense",
                 "classification", "summarization", "vision"

    Returns:
        A LangChain BaseChatModel instance configured per system settings.
        Includes TokenTrackingCallback for automatic usage tracking.
    """
    from usage_tracker import TokenTrackingCallback

    config = get_system_config(f"llm.{purpose}")
    provider = config.get("provider", "openai")
    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.2)
    callbacks = [TokenTrackingCallback(model=model, provider=provider, purpose=purpose)]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, callbacks=callbacks)

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, callbacks=callbacks)

    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=temperature, callbacks=callbacks)
        except ImportError:
            print("[model_factory] langchain-google-genai not installed, falling back to OpenAI")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, callbacks=callbacks)

    else:
        # Unknown provider → fallback to OpenAI
        print(f"[model_factory] Unknown provider '{provider}', falling back to OpenAI")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, callbacks=callbacks)


def invalidate_cache() -> None:
    """Clear the config cache, forcing a fresh DB read on next call."""
    _CONFIG_CACHE.clear()
    _CACHE_TIMESTAMPS.clear()
