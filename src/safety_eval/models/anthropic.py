"""Anthropic Messages API adapter."""

from __future__ import annotations

import time
from typing import Any

import anthropic
from anthropic.types import TextBlock

from safety_eval.models.base import (
    AdapterError,
    FatalAdapterError,
    ModelAdapter,
    RateLimitError,
    TransientError,
)
from safety_eval.types import FinishReason, GenerationParams, ModelResponse

_FINISH = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "model_context_window_exceeded": FinishReason.LENGTH,
    # A safety-classifier decline (HTTP 200). For an eval this is the model refusing,
    # so it is a response, not an error.
    "refusal": FinishReason.CONTENT_FILTER,
}


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str,
        model: str,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(model_id)
        self.model = model
        # The runner owns retries, so the SDK's own retry loop is turned off.
        self.client = client or anthropic.AsyncAnthropic(max_retries=0, timeout=300.0)

    async def generate(
        self,
        prompt: str,
        system: str | None,
        params: GenerationParams,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {}
        if system is not None:
            kwargs["system"] = system
        if params.temperature is not None:
            # SDK 1.x dropped the sampling kwargs. Older models (e.g. Haiku 4.5) still
            # accept temperature; Opus 4.7+ reject it with a 400.
            kwargs["extra_body"] = {"temperature": params.temperature}

        start = time.perf_counter()
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=params.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise FatalAdapterError(f"{self.model_id}: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise FatalAdapterError(f"{self.model_id}: model not found ({e.message})") from e
        except anthropic.InternalServerError as e:
            raise TransientError(str(e)) from e
        except anthropic.APIStatusError as e:
            if e.status_code in (408, 409) or e.status_code >= 500:  # incl. 529 overloaded
                raise TransientError(str(e)) from e
            raise AdapterError(f"HTTP {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:  # includes timeouts
            raise TransientError(str(e)) from e

        text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        stop = message.stop_reason
        return ModelResponse(
            model=self.model_id,
            text=text,
            finish_reason=_FINISH.get(stop or "", FinishReason.STOP),
            model_version=message.model,
            stop_reason=stop,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def aclose(self) -> None:
        await self.client.close()
