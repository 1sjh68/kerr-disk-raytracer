from __future__ import annotations

from src.config import load_config
from src.render import (
    metadata_from_render,
    render_thin_disk_geodesic_cpu,
    save_maps_npz,
    save_png,
    upscale_rgb,
    write_json,
)


def main() -> None:
    cfg = load_config()
    resolution = int(cfg["render"].get("geodesic_resolution", 32))
    data = render_thin_disk_geodesic_cpu(cfg, resolution=resolution)
    preview = upscale_rgb(data["rgb"])
    save_png("output/cpu_geodesic_image.png", preview)
    save_png("figures/geodesic_reference.png", preview)
    save_maps_npz("reference/cpu_geodesic_reference.npz", data)
    write_json("logs/cpu_geodesic_run.json", metadata_from_render(data))


if __name__ == "__main__":
    main()
