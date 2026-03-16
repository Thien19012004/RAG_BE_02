"""
usage_tracker.py — Thread-safe LLM/Embedding token usage tracker.

Uses ContextVar to accumulate token counts per-request.
Attach TokenTrackingCallback to LLM calls via model_factory.get_llm().
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class UsageRecord:
    """A single LLM/Embedding usage record."""
    model: str = "unknown"
    provider: str = "unknown"
    purpose: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class RequestUsage:
    """Accumulated usage for one API request."""
    records: List[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    def total_input(self) -> int:
        return sum(r.input_tokens for r in self.records)

    def total_output(self) -> int:
        return sum(r.output_tokens for r in self.records)

    def clear(self) -> None:
        self.records.clear()


# Per-request accumulator
_request_usage: ContextVar[Optional[RequestUsage]] = ContextVar(
    "request_usage", default=None
)


def start_tracking() -> RequestUsage:
    """Start tracking token usage for the current request."""
    usage = RequestUsage()
    _request_usage.set(usage)
    return usage


def get_current_usage() -> Optional[RequestUsage]:
    """Get the current request's usage accumulator."""
    return _request_usage.get(None)


def stop_tracking() -> Optional[RequestUsage]:
    """Stop tracking and return accumulated usage."""
    usage = _request_usage.get(None)
    _request_usage.set(None)
    return usage


class TokenTrackingCallback(BaseCallbackHandler):
    """
    LangChain callback that captures token usage from LLM responses.
    Works with OpenAI, Groq, and Gemini providers.
    """

    def __init__(self, model: str, provider: str, purpose: str):
        self.model = model
        self.provider = provider
        self.purpose = purpose

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = get_current_usage()
        if usage is None:
            return

        # Extract token counts from LLM result
        input_tokens = 0
        output_tokens = 0

        llm_output = response.llm_output or {}

        # OpenAI / Groq format
        token_usage = llm_output.get("token_usage", {})
        if token_usage:
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

        # Also check per-generation metadata (some providers put it here)
        if input_tokens == 0 and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    info = getattr(gen, "generation_info", None) or {}
                    if "token_usage" in info:
                        input_tokens += info["token_usage"].get("prompt_tokens", 0)
                        output_tokens += info["token_usage"].get("completion_tokens", 0)

        usage.add(UsageRecord(
            model=self.model,
            provider=self.provider,
            purpose=self.purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ))


def track_embedding_usage(
    model: str,
    provider: str,
    num_texts: int,
) -> None:
    """
    Log an embedding call to the current request's tracker.
    Token count is estimated as ~tokens_per_text * num_texts.
    Exact count unavailable for OpenAI embeddings API.
    """
    usage = get_current_usage()
    if usage is None:
        return

    usage.add(UsageRecord(
        model=model,
        provider=provider,
        purpose="embedding",
        input_tokens=0,  # OpenAI embedding API doesn't return token counts
        output_tokens=0,
    ))
