from __future__ import annotations

from src.config import load_config
from src.render import (
    metadata_from_render,
    render_thin_disk_fast,
    save_maps_npz,
    save_png,
    write_json,
)


def main() -> None:
    cfg = load_config()
    data = render_thin_disk_fast(cfg)
    save_png("output/cpu_image.png", data["rgb"])
    save_maps_npz("reference/cpu_reference.npz", data)
    write_json("logs/cpu_run.json", metadata_from_render(data))

    cfg256 = load_config()
    data256 = render_thin_disk_fast(cfg256, resolution=256)
    save_maps_npz("reference/cpu_reference_256.npz", data256)
    write_json("logs/cpu_run_256.json", metadata_from_render(data256))


if __name__ == "__main__":
    main()
