"""Command-line entry point."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import NoReturn

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from safety_eval import __version__
from safety_eval.config import SuiteSpec
from safety_eval.models.base import FatalAdapterError, ModelAdapter
from safety_eval.models.registry import PROVIDERS, UnknownModelError, create_adapter, parse_model_id
from safety_eval.runner.cache import DEFAULT_CACHE_DIR, ResponseCache
from safety_eval.runner.executor import RunConfig, RunStats, run_suite
from safety_eval.suites.loader import SuiteError, list_builtin, load_suite
from safety_eval.types import GenerationParams

app = typer.Typer(help="Run reproducible safety evaluations against LLMs.", no_args_is_help=True)
console = Console()
err = Console(stderr=True)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def run(
    suite: str = typer.Option(..., help="Built-in suite name or path to a YAML suite."),
    model: str = typer.Option(..., help="Model id, e.g. anthropic:claude-haiku-4-5."),
    out: Path | None = typer.Option(
        None, help="JSONL output path. Default: results/<suite>__<model>.jsonl"
    ),
    repeats: int = typer.Option(1, min=1, help="Times to run each case (for error bars)."),
    concurrency: int = typer.Option(4, min=1, help="Max requests in flight."),
    max_tokens: int = typer.Option(4096, min=1, help="Max output tokens per response."),
    temperature: float | None = typer.Option(
        None, help="Only sent if set; some models reject any temperature."
    ),
    max_retries: int = typer.Option(5, min=0, help="Retries on rate limits and 5xx."),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, help="Response cache directory."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore and don't write the cache."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print prompts without calling any API."),
) -> None:
    """Run a suite against a model and append results to a JSONL file.

    Re-running with the same --out resumes: finished (case, repeat) pairs are skipped.
    """
    load_dotenv()
    spec = _load_suite_or_exit(suite)
    try:
        parse_model_id(model)
    except UnknownModelError as e:
        _fail(str(e))

    if dry_run:
        _print_dry_run(spec, model, repeats)
        return

    out = out or Path("results") / f"{spec.name}__{_slug(model)}.jsonl"
    params = GenerationParams(temperature=temperature, max_tokens=max_tokens)
    config = RunConfig(repeats=repeats, concurrency=concurrency, max_retries=max_retries)
    cache = None if no_cache else ResponseCache(cache_dir)

    try:
        adapter = create_adapter(model)
    except Exception as e:  # SDK constructors raise their own types, e.g. a missing API key
        _fail(f"could not create adapter for {model}: {e}")

    try:
        stats = asyncio.run(_run_with_progress(adapter, spec, params, out, config, cache))
    except FatalAdapterError as e:
        _fail(f"run stopped: {e}\nPartial results (if any) are in {out}; rerun to resume.")
    _print_summary(stats, out)


async def _run_with_progress(
    adapter: ModelAdapter,
    spec: SuiteSpec,
    params: GenerationParams,
    out: Path,
    config: RunConfig,
    cache: ResponseCache | None,
) -> RunStats:
    columns = (
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=err, transient=False) as progress:
        task = progress.add_task(f"{spec.name} · {adapter.model_id}", total=None)
        try:
            return await run_suite(
                adapter,
                spec,
                params,
                out,
                config,
                cache,
                on_start=lambda n: progress.update(task, total=n),
                on_result=lambda _: progress.advance(task),
            )
        finally:
            await adapter.aclose()


@app.command("list-suites")
def list_suites() -> None:
    """List built-in suites."""
    table = Table("name", "category", "cases", "description")
    for name in list_builtin():
        spec = _load_suite_or_exit(name)
        table.add_row(spec.name, spec.category.value, str(len(spec.cases)), spec.description or "")
    console.print(table)


@app.command("list-models")
def list_models() -> None:
    """List supported providers and the model id format for each."""
    table = Table("provider", "example id", "reads")
    for p in PROVIDERS.values():
        table.add_row(p.name, p.example, p.env or "")
    console.print(table)
    console.print("Model ids are provider:model, where model is the provider's own model name.")


def _print_dry_run(spec: SuiteSpec, model: str, repeats: int) -> None:
    cases = spec.test_cases()
    console.print(
        f"[bold]{spec.name}[/] v{spec.version} · {len(cases)} cases x {repeats} repeat(s) "
        f"→ {len(cases) * repeats} requests to {model}"
    )
    for c in cases:
        expect = "refuse" if c.should_refuse else "answer"
        console.print(f"\n[cyan]{c.id}[/] (expect: {expect})")
        if c.system:
            console.print(f"  [dim]system:[/] {c.system}")
        console.print(f"  {c.prompt}", markup=False)


def _print_summary(stats: RunStats, out: Path) -> None:
    console.print(
        f"Wrote {stats.completed} result(s) to {out} "
        f"({stats.resumed} already done, {stats.cache_hits} from cache, {stats.errors} error(s))."
    )
    console.print(
        f"Tokens: {stats.input_tokens:,} in / {stats.output_tokens:,} out (cache hits excluded)."
    )
    if stats.errors:
        console.print("[yellow]Some cases errored; rerun the same command to retry just those.[/]")


def _load_suite_or_exit(name_or_path: str) -> SuiteSpec:
    try:
        return load_suite(name_or_path)
    except SuiteError as e:
        _fail(str(e))


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def _fail(message: str) -> NoReturn:
    err.print(f"[red]error:[/] {escape(message)}", highlight=False)
    raise typer.Exit(code=2)
