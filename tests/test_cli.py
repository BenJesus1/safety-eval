from pathlib import Path

import pytest
from typer.testing import CliRunner

from safety_eval.cli import app
from safety_eval.types import RunResult

runner = CliRunner()


def test_dry_run_prints_prompts_without_calling_models() -> None:
    result = runner.invoke(
        app, ["run", "--suite", "over_refusal", "--model", "anthropic:x", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "or-001" in result.output and "expect: answer" in result.output


def test_run_with_fake_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out.jsonl"
    args = ["run", "--suite", "over_refusal", "--model", "fake:echo", "--out", str(out)]
    result = runner.invoke(app, [*args, "--repeats", "2"])
    assert result.exit_code == 0, result.output
    lines = out.read_text().splitlines()
    assert len(lines) == 8
    assert RunResult.from_jsonl(lines[0]).response.model == "fake:echo"
    assert (tmp_path / ".safety_eval_cache").is_dir()


def test_bad_inputs_exit_with_code_2() -> None:
    bad_model = runner.invoke(app, ["run", "--suite", "over_refusal", "--model", "gpt"])
    bad_suite = runner.invoke(app, ["run", "--suite", "nope", "--model", "fake:echo"])
    assert bad_model.exit_code == bad_suite.exit_code == 2


def test_list_commands() -> None:
    suites = runner.invoke(app, ["list-suites"])
    models = runner.invoke(app, ["list-models"])
    assert suites.exit_code == models.exit_code == 0
    assert "over_refusal" in suites.output and "anthropic" in models.output
