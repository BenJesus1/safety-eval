"""Map model id strings like "anthropic:claude-haiku-4-5" to adapters."""

from __future__ import annotations

from dataclasses import dataclass

from safety_eval.models.base import ModelAdapter


@dataclass(frozen=True)
class Provider:
    name: str
    example: str
    env: str | None  # credential or config the provider reads


PROVIDERS = {
    p.name: p
    for p in [
        Provider("anthropic", "anthropic:claude-haiku-4-5", "ANTHROPIC_API_KEY"),
        Provider("openai", "openai:<model-id>", "OPENAI_API_KEY"),
        Provider("ollama", "ollama:gemma4:e2b", "OLLAMA_HOST (optional)"),
        Provider("fake", "fake:echo", None),
    ]
}


class UnknownModelError(ValueError):
    pass


def parse_model_id(model_id: str) -> tuple[str, str]:
    """Split "provider:model". Only the first colon splits, so Ollama tags survive."""
    provider, sep, model = model_id.partition(":")
    if not sep or not model:
        raise UnknownModelError(f"model id must look like provider:model, got {model_id!r}")
    if provider not in PROVIDERS:
        raise UnknownModelError(f"unknown provider {provider!r} (known: {', '.join(PROVIDERS)})")
    return provider, model


def create_adapter(model_id: str) -> ModelAdapter:
    provider, model = parse_model_id(model_id)
    # SDKs are imported lazily so e.g. a dry run needs no API keys.
    if provider == "anthropic":
        from safety_eval.models.anthropic import AnthropicAdapter

        return AnthropicAdapter(model_id, model)
    if provider == "openai":
        from safety_eval.models.openai import OpenAIAdapter

        return OpenAIAdapter(model_id, model)
    if provider == "ollama":
        from safety_eval.models.ollama import OllamaAdapter

        return OllamaAdapter(model_id, model)
    from safety_eval.models.fake import FakeAdapter

    return FakeAdapter(model_id)
