import asyncio
import contextlib
from pathlib import Path

import pytest

from safety_eval.config import CaseSpec, SuiteSpec
from safety_eval.models.base import (
    AdapterError,
    FatalAdapterError,
    ModelAdapter,
    RateLimitError,
    TransientError,
)
from safety_eval.models.fake import FakeAdapter
from safety_eval.runner.cache import ResponseCache
from safety_eval.runner.executor import RunConfig, run_suite
from safety_eval.types import (
    Category,
    FinishReason,
    GenerationParams,
    ModelResponse,
    RunResult,
)

SUITE = SuiteSpec(
    name="mini",
    category=Category.JAILBREAK,
    cases=[CaseSpec(id=f"c{i}", prompt=f"prompt {i}", should_refuse=True) for i in range(5)],
)
PARAMS = GenerationParams()
FAST = RunConfig(backoff_base=0.0)


def read(out: Path) -> list[RunResult]:
    return [RunResult.from_jsonl(line) for line in out.read_text().splitlines()]


class ScriptedAdapter(ModelAdapter):
    """Raises the queued errors in order, then answers."""

    def __init__(self, errors: list[Exception]) -> None:
        super().__init__("fake:scripted")
        self.errors = errors
        self.calls = 0

    async def generate(
        self, prompt: str, system: str | None, params: GenerationParams
    ) -> ModelResponse:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return ModelResponse(model=self.model_id, text="ok")


async def test_writes_one_line_per_case_and_repeat(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    stats = await run_suite(FakeAdapter(), SUITE, PARAMS, out, RunConfig(repeats=3))
    results = read(out)
    assert stats.total == stats.completed == len(results) == 15
    assert {(r.case.id, r.sample) for r in results} == {
        (f"c{i}", s) for i in range(5) for s in range(3)
    }
    assert len({r.run_id for r in results}) == 1
    assert stats.input_tokens > 0


async def test_resume_skips_finished_pairs(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    first = await run_suite(FakeAdapter(), SUITE, PARAMS, out, RunConfig(repeats=2))
    # Simulate a crash: drop two finished lines and leave a truncated one.
    lines = out.read_text().splitlines()[:-2]
    out.write_text("\n".join(lines) + "\n" + lines[0][:20])

    adapter = FakeAdapter()
    stats = await run_suite(adapter, SUITE, PARAMS, out, RunConfig(repeats=2))
    assert (stats.resumed, stats.completed, adapter.calls) == (8, 2, 2)
    run_ids = {r.run_id for r in read_valid(out)}
    assert len(run_ids) == 1 and first.total == 10


def read_valid(out: Path) -> list[RunResult]:
    results = []
    for line in out.read_text().splitlines():
        with contextlib.suppress(ValueError):
            results.append(RunResult.from_jsonl(line))
    return results


async def test_cache_makes_reruns_free(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    await run_suite(FakeAdapter(), SUITE, PARAMS, tmp_path / "a.jsonl", cache=cache)
    adapter = FakeAdapter()
    stats = await run_suite(adapter, SUITE, PARAMS, tmp_path / "b.jsonl", cache=cache)
    assert adapter.calls == 0 and stats.cache_hits == 5
    assert all(r.response.cached for r in read(tmp_path / "b.jsonl"))
    assert (stats.input_tokens, stats.output_tokens) == (0, 0)


async def test_repeats_are_cached_separately(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    adapter = FakeAdapter()
    await run_suite(adapter, SUITE, PARAMS, tmp_path / "a.jsonl", RunConfig(repeats=2), cache)
    assert adapter.calls == 10


async def test_retries_transient_errors_then_succeeds(tmp_path: Path) -> None:
    one_case = SUITE.model_copy(update={"cases": SUITE.cases[:1]})
    adapter = ScriptedAdapter([RateLimitError("429"), TransientError("503")])
    await run_suite(adapter, one_case, PARAMS, tmp_path / "r.jsonl", FAST)
    [result] = read(tmp_path / "r.jsonl")
    assert adapter.calls == 3 and result.response.text == "ok"


async def test_gives_up_after_max_retries_and_retries_on_resume(tmp_path: Path) -> None:
    one_case = SUITE.model_copy(update={"cases": SUITE.cases[:1]})
    out = tmp_path / "r.jsonl"
    cache = ResponseCache(tmp_path / "cache")
    config = RunConfig(max_retries=2, backoff_base=0.0)
    adapter = ScriptedAdapter([TransientError("503")] * 3)
    stats = await run_suite(adapter, one_case, PARAMS, out, config, cache)
    [result] = read(out)
    assert adapter.calls == 3 and stats.errors == 1
    assert result.response.finish_reason is FinishReason.ERROR
    assert "TransientError after 3 attempt(s)" in (result.response.error or "")

    # Errors are neither cached nor treated as done: a rerun tries again.
    retry = ScriptedAdapter([])
    stats = await run_suite(retry, one_case, PARAMS, out, config, cache)
    assert retry.calls == 1 and stats.completed == 1 and stats.errors == 0


async def test_non_retryable_error_is_recorded_once(tmp_path: Path) -> None:
    one_case = SUITE.model_copy(update={"cases": SUITE.cases[:1]})
    adapter = ScriptedAdapter([AdapterError("HTTP 400: bad request")])
    await run_suite(adapter, one_case, PARAMS, tmp_path / "r.jsonl", FAST)
    assert adapter.calls == 1
    [result] = read(tmp_path / "r.jsonl")
    assert result.response.finish_reason is FinishReason.ERROR


async def test_fatal_error_stops_the_run(tmp_path: Path) -> None:
    adapter = ScriptedAdapter([FatalAdapterError("bad key")] * 10)
    with pytest.raises(FatalAdapterError):
        await run_suite(adapter, SUITE, PARAMS, tmp_path / "r.jsonl", RunConfig(concurrency=1))
    assert adapter.calls == 1


async def test_concurrency_limit_is_respected(tmp_path: Path) -> None:
    class Slow(ModelAdapter):
        def __init__(self) -> None:
            super().__init__("fake:slow")
            self.in_flight = self.peak = 0

        async def generate(
            self, prompt: str, system: str | None, params: GenerationParams
        ) -> ModelResponse:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            await asyncio.sleep(0.01)
            self.in_flight -= 1
            return ModelResponse(model=self.model_id, text="ok")

    adapter = Slow()
    config = RunConfig(repeats=4, concurrency=3)
    await run_suite(adapter, SUITE, PARAMS, tmp_path / "r.jsonl", config)
    assert adapter.peak == 3
