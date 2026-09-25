"""Command-line entry point. Commands are stubs until Phase 1."""

from __future__ import annotations

from pathlib import Path

import typer

from safety_eval import __version__

app = typer.Typer(help="Run reproducible safety evaluations against LLMs.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def run(
    suite: str = typer.Option(..., help="Built-in suite name or path to a YAML suite."),
    model: str = typer.Option(..., help="Model id, e.g. anthropic:claude-haiku-4-5."),
    out: Path = typer.Option(Path("results.jsonl"), help="JSONL output path."),
    dry_run: bool = typer.Option(False, help="Print prompts without calling any API."),
) -> None:
    """Run a suite against a model. (Phase 1)"""
    raise typer.Exit(code=_not_implemented("run"))


@app.command("list-suites")
def list_suites() -> None:
    """List built-in suites. (Phase 1)"""
    raise typer.Exit(code=_not_implemented("list-suites"))


def _not_implemented(name: str) -> int:
    typer.echo(f"`{name}` is not implemented yet.", err=True)
    return 2
