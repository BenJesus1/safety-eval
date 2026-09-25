"""Run a suite against one model: bounded concurrency, retries, caching, repeats and resume."""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from safety_eval.config import SuiteSpec
from safety_eval.models.base import (
    AdapterError,
    FatalAdapterError,
    ModelAdapter,
    RateLimitError,
    TransientError,
)
from safety_eval.runner.cache import ResponseCache
from safety_eval.types import (
    FinishReason,
    GenerationParams,
    ModelResponse,
    RunResult,
    TestCase,
    cache_key,
)


@dataclass(frozen=True)
class RunConfig:
    repeats: int = 1
    concurrency: int = 4
    max_retries: int = 5
    backoff_base: float = 1.0  # seconds; doubles each attempt
    backoff_max: float = 60.0


@dataclass
class RunStats:
    total: int = 0  # cases x repeats
    resumed: int = 0  # already in the output file, skipped
    completed: int = 0  # written this run
    cache_hits: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def read_completed(out: Path, suite: str, model_id: str) -> tuple[str | None, set[tuple[str, int]]]:
    """Return the existing run id and the (case id, sample) pairs already written to `out`.

    Error results don't count as done, so a rerun retries them; the file then holds
    both lines and readers should keep the last line per (case id, sample).
    A truncated last line (from a crash mid-write) is ignored, so that case is rerun.
    """
    if not out.exists():
        return None, set()
    run_id: str | None = None
    done: set[tuple[str, int]] = set()
    with out.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                result = RunResult.from_jsonl(line)
            except (ValidationError, json.JSONDecodeError):
                continue
            if result.suite == suite and result.response.model == model_id:
                run_id = run_id or result.run_id
                if result.response.finish_reason is not FinishReason.ERROR:
                    done.add((result.case.id, result.sample))
    return run_id, done


async def run_suite(
    adapter: ModelAdapter,
    suite: SuiteSpec,
    params: GenerationParams,
    out: Path,
    config: RunConfig | None = None,
    cache: ResponseCache | None = None,
    on_start: Callable[[int], None] | None = None,
    on_result: Callable[[RunResult], None] | None = None,
) -> RunStats:
    """Append one RunResult per (case, sample) to `out`, skipping pairs already there.

    Raises FatalAdapterError (after writing whatever finished) if credentials or the
    model id are bad, since every remaining request would fail the same way.
    """
    config = config or RunConfig()
    cases = suite.test_cases()
    existing_run_id, done = read_completed(out, suite.name, adapter.model_id)
    run_id = existing_run_id or uuid.uuid4().hex[:12]
    work = [(c, s) for c in cases for s in range(config.repeats) if (c.id, s) not in done]

    stats = RunStats(total=len(cases) * config.repeats)
    stats.resumed = stats.total - len(work)
    if on_start:
        on_start(len(work))

    semaphore = asyncio.Semaphore(config.concurrency)
    write_lock = asyncio.Lock()
    stop = asyncio.Event()  # set on a fatal error so queued cases never start
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("a", encoding="utf-8") as fh:

        async def one(case: TestCase, sample: int) -> None:
            async with semaphore:
                if stop.is_set():
                    return
                try:
                    response = await _generate(adapter, case, params, sample, config, cache)
                except FatalAdapterError:
                    stop.set()
                    raise
            result = RunResult(
                run_id=run_id,
                suite=suite.name,
                case=case,
                params=params,
                response=response,
                sample=sample,
            )
            async with write_lock:
                fh.write(result.to_jsonl())
                fh.flush()
                stats.completed += 1
                stats.cache_hits += response.cached
                stats.errors += response.finish_reason is FinishReason.ERROR
                if not response.cached:
                    stats.input_tokens += response.input_tokens
                    stats.output_tokens += response.output_tokens
            if on_result:
                on_result(result)

        tasks = [asyncio.create_task(one(c, s)) for c, s in work]
        try:
            await asyncio.gather(*tasks)
        except FatalAdapterError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    return stats


async def _generate(
    adapter: ModelAdapter,
    case: TestCase,
    params: GenerationParams,
    sample: int,
    config: RunConfig,
    cache: ResponseCache | None,
) -> ModelResponse:
    key = cache_key(adapter.model_id, case, params, sample)
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit

    for attempt in range(config.max_retries + 1):
        try:
            response = await adapter.generate(case.prompt, case.system, params)
        except FatalAdapterError:
            raise
        except (RateLimitError, TransientError) as e:
            if attempt == config.max_retries:
                return _error_response(adapter, e, attempts=attempt + 1)
            delay = min(config.backoff_max, config.backoff_base * 2**attempt)
            await asyncio.sleep(delay * random.uniform(0.5, 1.0))  # jitter spreads out retries
        except AdapterError as e:
            return _error_response(adapter, e, attempts=attempt + 1)
        else:
            if cache is not None:
                cache.put(key, response)
            return response
    raise AssertionError("unreachable")


def _error_response(adapter: ModelAdapter, error: Exception, attempts: int) -> ModelResponse:
    return ModelResponse(
        model=adapter.model_id,
        text="",
        finish_reason=FinishReason.ERROR,
        error=f"{type(error).__name__} after {attempts} attempt(s): {error}",
    )
