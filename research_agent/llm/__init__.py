"""LLM provider factory. Swap providers with LLM_PROVIDER=anthropic|groq."""
from __future__ import annotations

from research_agent.config import settings
from research_agent.llm.base import LLMClient, LLMResponse, ToolCall, Usage  # noqa: F401


def get_llm(provider: str | None = None, strong: bool = False) -> LLMClient:
    provider = provider or settings.llm_provider
    if provider == "anthropic":
        from research_agent.llm.anthropic_client import AnthropicClient

        model = (settings.anthropic_model_strong if strong and settings.anthropic_model_strong
                 else settings.anthropic_model)
        return AnthropicClient(model)
    if provider == "groq":
        from research_agent.llm.groq_client import GroqClient

        return GroqClient(settings.groq_model)
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}")
