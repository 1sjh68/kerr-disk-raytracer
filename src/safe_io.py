from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MAX_TEXT_BYTES = 10 * 1024 * 1024
MAX_JSON_BYTES = 5 * 1024 * 1024


def read_limited_text(
    path: str | Path,
    *,
    max_bytes: int = MAX_TEXT_BYTES,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> str:
    resolved = Path(path)
    size = resolved.stat().st_size
    if size > max_bytes:
        raise ValueError(f"refusing to read oversized file: {resolved} ({size} bytes)")
    return resolved.read_text(encoding=encoding, errors=errors)


def read_limited_json(path: str | Path, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    try:
        return json.loads(read_limited_text(path, max_bytes=max_bytes))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
