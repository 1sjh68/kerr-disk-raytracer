from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.gpu_trace import render_gpu_or_fallback
from src.render import metadata_from_render, save_maps_npz, save_png, write_json


def main() -> None:
    cfg = load_config()
    data = render_gpu_or_fallback(cfg)
    save_png("output/gpu_image.png", data["rgb"])
    legacy_pickle = Path("output/gpu_data.npy")
    if legacy_pickle.exists():
        legacy_pickle.unlink()
    save_maps_npz("output/gpu_data.npz", data)
    meta = metadata_from_render(data)
    meta["backend"] = data["backend"]
    meta["backend_reason"] = data["backend_reason"]
    write_json("logs/gpu_run.json", meta)


if __name__ == "__main__":
    main()
