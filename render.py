from __future__ import annotations

from src.config import load_config
from src.render import render_thin_disk_fast, save_png, save_maps_npz, write_json, metadata_from_render


def main() -> None:
    cfg = load_config()
    data = render_thin_disk_fast(cfg, resolution=256)
    save_png("figures/final_render.png", data["rgb"])
    save_maps_npz("output/final_render_maps.npz", data)
    write_json("logs/render_run.json", metadata_from_render(data))


if __name__ == "__main__":
    main()

