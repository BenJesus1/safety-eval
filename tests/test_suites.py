from pathlib import Path

import pytest

from safety_eval.suites.loader import SuiteError, list_builtin, load_suite
from safety_eval.types import Category


def write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "suite.yaml"
    path.write_text(text)
    return str(path)


def test_builtin_over_refusal_loads() -> None:
    assert "over_refusal" in list_builtin()
    spec = load_suite("over_refusal")
    assert spec.category is Category.OVER_REFUSAL
    cases = spec.test_cases()
    assert cases and all(not c.should_refuse for c in cases)


def test_case_system_overrides_suite_system(tmp_path: Path) -> None:
    spec = load_suite(
        write(
            tmp_path,
            """\
name: s
category: jailbreak
system: suite-level
cases:
  - id: a
    prompt: p
    should_refuse: true
  - id: b
    prompt: p
    should_refuse: true
    system: case-level
""",
        )
    )
    a, b = spec.test_cases()
    assert (a.system, b.system) == ("suite-level", "case-level")


def test_validation_error_points_at_line(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """\
name: s
category: over_refusal
cases:
  - id: a
    prompt: fine
    should_refuse: false
  - id: b
    prompt: missing should_refuse
  - id: c
    prompt: p
    should_refuse: maybe
""",
    )
    with pytest.raises(SuiteError) as exc:
        load_suite(path)
    msg = str(exc.value)
    assert f"{path}:7: cases.1.should_refuse: Field required" in msg
    assert f"{path}:11: cases.2.should_refuse:" in msg


def test_unknown_field_and_bad_category(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "name: s\ncategory: nope\ncases:\n  - {id: a, prompt: p, should_refuse: true, oops: 1}\n",
    )
    with pytest.raises(SuiteError) as exc:
        load_suite(path)
    msg = str(exc.value)
    assert f"{path}:2: category:" in msg
    assert "cases.0.oops: Extra inputs are not permitted" in msg


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "name: s\ncategory: jailbreak\ncases:\n"
        "  - {id: a, prompt: p, should_refuse: true}\n"
        "  - {id: a, prompt: q, should_refuse: true}\n",
    )
    with pytest.raises(SuiteError, match="duplicate case ids: a"):
        load_suite(path)


def test_invalid_yaml_and_unknown_suite(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="invalid YAML"):
        load_suite(write(tmp_path, "name: [unclosed\n"))
    with pytest.raises(SuiteError, match=r"built-ins: .*over_refusal"):
        load_suite("does_not_exist")
