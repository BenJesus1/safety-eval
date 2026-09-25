"""Find and load YAML suites, with validation errors that point at the offending line."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from safety_eval.config import SuiteSpec

BUILTIN_DIR = Path(__file__).parent / "builtin"


class SuiteError(Exception):
    """A suite could not be found, parsed or validated."""


def list_builtin() -> list[str]:
    return sorted(p.stem for p in BUILTIN_DIR.glob("*.yaml"))


def resolve_suite(name_or_path: str) -> Path:
    """A path to an existing file wins; otherwise look up a built-in suite by name."""
    path = Path(name_or_path)
    if path.is_file():
        return path
    builtin = BUILTIN_DIR / f"{name_or_path}.yaml"
    if builtin.is_file():
        return builtin
    available = ", ".join(list_builtin()) or "none"
    raise SuiteError(
        f"no suite file or built-in suite named {name_or_path!r} (built-ins: {available})"
    )


def load_suite(name_or_path: str) -> SuiteSpec:
    path = resolve_suite(name_or_path)
    text = path.read_text(encoding="utf-8")
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SuiteError(f"{path}: invalid YAML: {e}") from e

    try:
        return SuiteSpec.model_validate(data)
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = err["loc"]
            where = ".".join(str(part) for part in loc) or "<suite>"
            lines.append(f"{path}:{_line_for(root, loc)}: {where}: {err['msg']}")
        raise SuiteError("\n".join(lines)) from e


def _line_for(node: yaml.Node | None, loc: tuple[int | str, ...]) -> int:
    """1-based line of the deepest YAML node `loc` reaches (the parent if a key is missing)."""
    if node is None:
        return 1
    line = node.start_mark.line
    for part in loc:
        child: yaml.Node | None = None
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if isinstance(key, yaml.ScalarNode) and key.value == part:
                    child = value
                    break
        elif (
            isinstance(node, yaml.SequenceNode)
            and isinstance(part, int)
            and 0 <= part < len(node.value)
        ):
            child = node.value[part]
        if child is None:
            break
        node = child
        line = node.start_mark.line
    return line + 1
