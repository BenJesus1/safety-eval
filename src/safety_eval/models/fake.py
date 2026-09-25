"""Deterministic offline adapter for tests and --dry-run style development."""

from __future__ import annotations

from safety_eval.models.base import ModelAdapter
from safety_eval.types import GenerationParams, ModelResponse

REFUSAL_TEXT = "I can't help with that."


class FakeAdapter(ModelAdapter):
    """Refuses any prompt containing one of `refuse_on`, otherwise echoes a canned answer."""

    def __init__(self, model_id: str = "fake:echo", refuse_on: tuple[str, ...] = ()) -> None:
        super().__init__(model_id)
        self.refuse_on = tuple(s.lower() for s in refuse_on)
        self.calls = 0

    async def generate(
        self,
        prompt: str,
        system: str | None,
        params: GenerationParams,
    ) -> ModelResponse:
        self.calls += 1
        lowered = prompt.lower()
        text = (
            REFUSAL_TEXT
            if any(s in lowered for s in self.refuse_on)
            else f"Sure. Here is a response to: {prompt[:80]}"
        )
        return ModelResponse(
            model=self.model_id,
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
        )
