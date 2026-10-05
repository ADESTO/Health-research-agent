"""LLM provider factory. Swap providers with LLM_PROVIDER=anthropic|groq|deepseek.

Model routing: each step of a run can use its own provider and model, set in MODEL_ROUTES, e.g.

    MODEL_ROUTES=cheap=deepseek:deepseek-flash; strong=anthropic:claude-sonnet-4-5; synthesis=anthropic:claude-opus-4-1

A step's own route wins over its tier (cheap or strong); anything not routed uses LLM_PROVIDER and its model.
"""
from __future__ import annotations

from research_agent.config import settings
from research_agent.llm.base import LLMClient, LLMResponse, ToolCall, Usage  # noqa: F401

# High-volume, well-specified steps: a cheap, fast model does them well.
CHEAP_STEPS = {"extraction", "protocol_fields", "recheck", "followup_reading", "number_check", "discovery",
               "literature", "evidence", "fieldpass"}
# Judgement and writing: worth a stronger model.
STRONG_STEPS = {"synthesis", "orchestrator", "gap_reasoning", "design", "design_plan", "design_writer", "followup", "contradictions",
                "draft_plan", "draft", "research", "critic", "research_supervisor"}


def parse_routes(text: str) -> dict[str, tuple[str, str]]:
    """'cheap=deepseek:deepseek-flash; synthesis=anthropic:claude-x' -> {name: (provider, model)}."""
    routes = {}
    for part in (text or "").replace("\n", ";").split(";"):
        if "=" not in part:
            continue
        name, target = (x.strip() for x in part.split("=", 1))
        provider, _, model = target.partition(":")
        if name and provider:
            routes[name.lower()] = (provider.strip().lower(), model.strip())
    return routes


def route_for(step: str | None, strong: bool = False) -> tuple[str, str] | None:
    routes = parse_routes(settings.model_routes)
    if not routes:
        return None
    if step and step.lower() in routes:
        return routes[step.lower()]
    tier = "strong" if (strong or (step or "") in STRONG_STEPS) else ("cheap" if (step or "") in CHEAP_STEPS else None)
    return routes.get(tier) if tier else None


def get_llm(provider: str | None = None, strong: bool = False, step: str | None = None,
            model: str | None = None) -> LLMClient:
    route = route_for(step, strong) if not model else None
    if route:
        provider, model = route[0], route[1] or None
    provider = provider or settings.llm_provider
    if provider == "anthropic":
        from research_agent.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model or (settings.anthropic_model_strong if strong and settings.anthropic_model_strong
                                         else settings.anthropic_model))
    if provider == "groq":
        from research_agent.llm.groq_client import GroqClient

        return GroqClient(model or settings.groq_model)
    if provider == "deepseek":
        from research_agent.llm.deepseek_client import DeepSeekClient

        return DeepSeekClient(model or (settings.deepseek_model_strong if strong and settings.deepseek_model_strong
                                        else settings.deepseek_model))
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}")
