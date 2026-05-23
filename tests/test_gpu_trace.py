from __future__ import annotations

import numpy as np

from src.config import load_config
from src.gpu_trace import cuda_available, render_gpu_or_fallback


def test_cuda_probe_returns_reason_string() -> None:
    available, reason = cuda_available()
    assert isinstance(available, bool)
    assert isinstance(reason, str)
    assert reason


def test_gpu_entrypoint_preserves_output_contract() -> None:
    cfg = load_config()
    data = render_gpu_or_fallback(cfg, resolution=8)
    assert data["intensity"].shape == (8, 8)
    assert data["redshift"].shape == (8, 8)
    assert data["temperature"].shape == (8, 8)
    assert data["hit_mask"].shape == (8, 8)
    assert data["rgb"].shape == (8, 8, 3)
    assert data["backend"] in {"cuda", "cuda_unknown", "cpu_fallback"}
    assert np.all(np.isfinite(data["intensity"]))
