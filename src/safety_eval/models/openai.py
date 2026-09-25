"""OpenAI Chat Completions adapter."""

from __future__ import annotations

import time
from typing import Any

import openai
from openai.types.chat import ChatCompletionMessageParam

from safety_eval.models.base import (
    AdapterError,
    FatalAdapterError,
    ModelAdapter,
    RateLimitError,
    TransientError,
)
from safety_eval.types import FinishReason, GenerationParams, ModelResponse

_FINISH = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class OpenAIAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str,
        model: str,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(model_id)
        self.model = model
        self.client = client or openai.AsyncOpenAI(max_retries=0, timeout=300.0)

    async def generate(
        self,
        prompt: str,
        system: str | None,
        params: GenerationParams,
    ) -> ModelResponse:
        messages: list[ChatCompletionMessageParam] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {}
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature

        start = time.perf_counter()
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=params.max_tokens,
                **kwargs,
            )
        except openai.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise FatalAdapterError(f"{self.model_id}: {e.message}") from e
        except openai.NotFoundError as e:
            raise FatalAdapterError(f"{self.model_id}: model not found ({e.message})") from e
        except openai.InternalServerError as e:
            raise TransientError(str(e)) from e
        except openai.APIStatusError as e:
            if e.status_code in (408, 409) or e.status_code >= 500:  # incl. 529 overloaded
                raise TransientError(str(e)) from e
            raise AdapterError(f"HTTP {e.status_code}: {e.message}") from e
        except openai.APIConnectionError as e:  # includes timeouts
            raise TransientError(str(e)) from e

        if not completion.choices:
            raise AdapterError(f"{self.model_id}: response had no choices")
        choice = completion.choices[0]
        text = choice.message.content or ""
        finish = _FINISH.get(choice.finish_reason, FinishReason.STOP)
        if not text and choice.message.refusal:
            text = choice.message.refusal
            finish = FinishReason.CONTENT_FILTER
        usage = completion.usage
        return ModelResponse(
            model=self.model_id,
            text=text,
            finish_reason=finish,
            model_version=completion.model,
            stop_reason=choice.finish_reason,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def aclose(self) -> None:
        await self.client.close()
