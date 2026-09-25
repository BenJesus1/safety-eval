"""Ollama adapter for local open-weights models (POST /api/chat)."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx2

from safety_eval.models.base import (
    AdapterError,
    FatalAdapterError,
    ModelAdapter,
    RateLimitError,
    TransientError,
)
from safety_eval.types import FinishReason, GenerationParams, ModelResponse

DEFAULT_HOST = "http://localhost:11434"

_FINISH = {"stop": FinishReason.STOP, "length": FinishReason.LENGTH}


class OllamaAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str,
        model: str,
        client: httpx2.AsyncClient | None = None,
    ) -> None:
        super().__init__(model_id)
        self.model = model
        host = os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
        # Local models can be slow on a laptop; a long read timeout avoids spurious retries.
        self.client = client or httpx2.AsyncClient(base_url=host, timeout=600.0)

    async def generate(
        self,
        prompt: str,
        system: str | None,
        params: GenerationParams,
    ) -> ModelResponse:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options: dict[str, Any] = {"num_predict": params.max_tokens}
        if params.temperature is not None:
            options["temperature"] = params.temperature
        body = {"model": self.model, "messages": messages, "stream": False, "options": options}

        start = time.perf_counter()
        try:
            resp = await self.client.post("/api/chat", json=body)
        except httpx2.ConnectError as e:
            raise FatalAdapterError(
                f"{self.model_id}: cannot reach Ollama at {self.client.base_url} "
                "(is `ollama serve` running?)"
            ) from e
        except httpx2.TransportError as e:  # timeouts and dropped connections
            raise TransientError(str(e)) from e

        if resp.status_code == 404:
            hint = f"try `ollama pull {self.model}`"
            raise FatalAdapterError(f"{self.model_id}: {resp.text} ({hint})")
        if resp.status_code == 429:
            raise RateLimitError(resp.text)
        if resp.status_code >= 500:
            raise TransientError(f"HTTP {resp.status_code}: {resp.text}")
        if resp.status_code >= 400:
            raise AdapterError(f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        done_reason = data.get("done_reason")
        return ModelResponse(
            model=self.model_id,
            text=data.get("message", {}).get("content", ""),
            finish_reason=_FINISH.get(done_reason or "", FinishReason.STOP),
            model_version=data.get("model"),
            stop_reason=done_reason,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def aclose(self) -> None:
        await self.client.aclose()
