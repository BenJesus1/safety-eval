"""Model adapter interface. Each provider implements `generate`."""

from __future__ import annotations

from abc import ABC, abstractmethod

from safety_eval.types import GenerationParams, ModelResponse


class AdapterError(Exception):
    """Base class for normalized provider errors."""


class RateLimitError(AdapterError):
    """Provider returned 429; the runner should back off and retry."""


class TransientError(AdapterError):
    """Timeout or 5xx; safe to retry."""


class ModelAdapter(ABC):
    """Wraps one provider model behind a uniform async API.

    `model_id` is the full registry string, e.g. "anthropic:claude-haiku-4-5".
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None,
        params: GenerationParams,
    ) -> ModelResponse:
        """Return the model's response.

        Raise RateLimitError / TransientError for retryable failures. Content-filter
        blocks are not errors: return a ModelResponse with finish_reason=CONTENT_FILTER.
        """
