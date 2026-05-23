from __future__ import annotations

import pytest

from src.safe_io import read_limited_json, read_limited_text


def test_read_limited_text_rejects_oversized_file(tmp_path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("abcdef", encoding="utf-8")

    with pytest.raises(ValueError):
        read_limited_text(path, max_bytes=5)


def test_read_limited_json_reports_invalid_json(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{bad", encoding="utf-8")

    with pytest.raises(ValueError):
        read_limited_json(path)
