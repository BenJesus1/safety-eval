"""Adapters against fake HTTP transports: request shape, response parsing, error mapping."""

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from safety_eval.models.anthropic import AnthropicAdapter
from safety_eval.models.base import FatalAdapterError, RateLimitError, TransientError
from safety_eval.models.ollama import OllamaAdapter
from safety_eval.models.openai import OpenAIAdapter
from safety_eval.models.registry import UnknownModelError, create_adapter, parse_model_id
from safety_eval.types import FinishReason, GenerationParams

Handler = Callable[[httpx2.Request], httpx2.Response]


class Recorder:
    """Returns a canned response and remembers the JSON body of each request."""

    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.status, self.body = status, body
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return httpx2.Response(self.status, json=self.body)


def http(handler: Handler, **kwargs: Any) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler), **kwargs)


def anthropic_adapter(handler: Handler) -> AnthropicAdapter:
    client = anthropic.AsyncAnthropic(api_key="test", max_retries=0, http_client=http(handler))
    return AnthropicAdapter("anthropic:claude-haiku-4-5", "claude-haiku-4-5", client)


def anthropic_message(stop_reason: str, text: str = "Hello") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


async def test_anthropic_request_and_response() -> None:
    rec = Recorder(200, anthropic_message("end_turn"))
    resp = await anthropic_adapter(rec).generate("hi", "be safe", GenerationParams(max_tokens=99))
    [body] = rec.requests
    assert body["system"] == "be safe" and body["max_tokens"] == 99
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in body
    assert (resp.text, resp.finish_reason, resp.model_version) == (
        "Hello",
        FinishReason.STOP,
        "claude-haiku-4-5",
    )
    assert (resp.input_tokens, resp.output_tokens) == (12, 3)


async def test_anthropic_sends_temperature_only_when_set() -> None:
    rec = Recorder(200, anthropic_message("end_turn"))
    await anthropic_adapter(rec).generate("hi", None, GenerationParams(temperature=0.0))
    assert rec.requests[0]["temperature"] == 0.0
    assert "system" not in rec.requests[0]


async def test_anthropic_refusal_is_a_response_not_an_error() -> None:
    rec = Recorder(200, anthropic_message("refusal", text=""))
    resp = await anthropic_adapter(rec).generate("hi", None, GenerationParams())
    assert resp.finish_reason is FinishReason.CONTENT_FILTER
    assert resp.stop_reason == "refusal" and resp.text == ""


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, RateLimitError), (529, TransientError), (401, FatalAdapterError)],
)
async def test_anthropic_error_mapping(status: int, expected: type[Exception]) -> None:
    body = {"type": "error", "error": {"type": "x", "message": "nope"}}
    with pytest.raises(expected):
        await anthropic_adapter(Recorder(status, body)).generate("hi", None, GenerationParams())


def openai_adapter(handler: Handler) -> OpenAIAdapter:
    client = openai.AsyncOpenAI(api_key="test", max_retries=0, http_client=http(handler))
    return OpenAIAdapter("openai:test-model", "test-model", client)


def openai_completion(
    finish: str, content: str | None, refusal: str | None = None
) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model-2026",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    }


async def test_openai_request_and_response() -> None:
    rec = Recorder(200, openai_completion("length", "partial"))
    resp = await openai_adapter(rec).generate("hi", "sys", GenerationParams(max_tokens=50))
    [body] = rec.requests
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["max_completion_tokens"] == 50 and "temperature" not in body
    assert (resp.text, resp.finish_reason) == ("partial", FinishReason.LENGTH)
    assert (resp.model_version, resp.input_tokens, resp.output_tokens) == ("test-model-2026", 7, 2)


async def test_openai_refusal_field() -> None:
    rec = Recorder(200, openai_completion("stop", None, refusal="I can't help with that."))
    resp = await openai_adapter(rec).generate("hi", None, GenerationParams())
    assert resp.finish_reason is FinishReason.CONTENT_FILTER
    assert resp.text == "I can't help with that."


async def test_openai_rate_limit() -> None:
    rec = Recorder(429, {"error": {"message": "slow down", "type": "rate_limit"}})
    with pytest.raises(RateLimitError):
        await openai_adapter(rec).generate("hi", None, GenerationParams())


def ollama_adapter(handler: Handler) -> OllamaAdapter:
    client = http(handler, base_url="http://ollama.test")
    return OllamaAdapter("ollama:gemma4:e2b", "gemma4:e2b", client)


async def test_ollama_request_and_response() -> None:
    rec = Recorder(
        200,
        {
            "model": "gemma4:e2b",
            "message": {"role": "assistant", "content": "Sure."},
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "eval_count": 4,
        },
    )
    resp = await ollama_adapter(rec).generate("hi", None, GenerationParams(temperature=0.5))
    [body] = rec.requests
    assert body["model"] == "gemma4:e2b" and body["stream"] is False
    assert body["options"] == {"num_predict": 4096, "temperature": 0.5}
    assert (resp.text, resp.input_tokens, resp.output_tokens) == ("Sure.", 20, 4)


async def test_ollama_missing_model_is_fatal() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, text='{"error":"model not found"}')

    with pytest.raises(FatalAdapterError, match="ollama pull gemma4:e2b"):
        await ollama_adapter(handler).generate("hi", None, GenerationParams())


def test_registry() -> None:
    assert parse_model_id("ollama:gemma4:e2b") == ("ollama", "gemma4:e2b")
    assert create_adapter("fake:echo").model_id == "fake:echo"
    for bad in ["claude-haiku-4-5", "nope:model", "anthropic:"]:
        with pytest.raises(UnknownModelError):
            parse_model_id(bad)
