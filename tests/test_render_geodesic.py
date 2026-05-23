from __future__ import annotations

import numpy as np

from src.config import load_config
from src.render import render_thin_disk_geodesic_cpu


def test_geodesic_renderer_produces_terminal_status_map() -> None:
    cfg = load_config()
    cfg["integration"]["max_steps"] = 160
    data = render_thin_disk_geodesic_cpu(cfg, resolution=6)
    assert data["intensity"].shape == (6, 6)
    assert data["status_code"].shape == (6, 6)
    assert np.any(data["status_code"] > 0)
    assert np.all(np.isfinite(data["null_error"]))
