from __future__ import annotations

import numpy as np

import run_gpu


def test_run_gpu_writes_non_pickle_npz_and_removes_legacy_npy(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    legacy = tmp_path / "output" / "gpu_data.npy"
    legacy.write_bytes(b"legacy pickle placeholder")

    def fake_render(_cfg):
        return {
            "intensity": np.ones((2, 2), dtype=np.float32),
            "redshift": np.ones((2, 2), dtype=np.float32),
            "temperature": np.ones((2, 2), dtype=np.float32),
            "hit_mask": np.ones((2, 2), dtype=np.uint8),
            "rgb": np.ones((2, 2, 3), dtype=np.float32),
            "elapsed_s": 0.01,
            "method": "test",
            "resolution": 2,
            "spin": 0.0,
            "inclination_deg": 60.0,
            "inner_radius": 6.0,
            "outer_radius": 28.0,
            "backend": "cpu_fallback",
            "backend_reason": "test",
        }

    monkeypatch.setattr(run_gpu, "load_config", lambda: {})
    monkeypatch.setattr(run_gpu, "render_gpu_or_fallback", fake_render)
    run_gpu.main()

    assert not legacy.exists()
    artifact = tmp_path / "output" / "gpu_data.npz"
    assert artifact.exists()
    with np.load(artifact, allow_pickle=False) as data:
        assert set(data.files) == {"intensity", "redshift", "temperature", "hit_mask", "rgb"}
