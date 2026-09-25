"""On-disk response cache: one JSON file per cache key, so reruns cost nothing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from safety_eval.types import ModelResponse

DEFAULT_CACHE_DIR = Path(".safety_eval_cache")


class ResponseCache:
    def __init__(self, root: Path = DEFAULT_CACHE_DIR) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> ModelResponse | None:
        path = self._path(key)
        try:
            response = ModelResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValidationError):
            return None
        return response.model_copy(update={"cached": True})

    def put(self, key: str, response: ModelResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash never leaves a half-written entry.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(response.model_copy(update={"cached": False}).model_dump_json())
        os.replace(tmp, path)
