from __future__ import annotations

import pytest

from src.config import load_config, validate_cuda_block, validate_resolution


def test_malicious_yaml_tag_is_rejected(tmp_path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text('!!python/object/apply:os.system ["echo unsafe"]\n', encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_unbounded_render_resolution_is_rejected(tmp_path) -> None:
    cfg_path = tmp_path / "huge.yaml"
    cfg_path.write_text("render:\n  resolution: 10000000\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_direct_resolution_override_is_bounded() -> None:
    with pytest.raises(ValueError):
        validate_resolution(10000000)


def test_invalid_cuda_block_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_cuda_block([0, 16])
    with pytest.raises(ValueError):
        validate_cuda_block([64, 64])

