from __future__ import annotations

import argparse
import numpy as np

from src.config import load_config
from src.gpu_trace import render_gpu_or_fallback_geodesic
from src.render import (
    metadata_from_render,
    save_maps_npz,
    save_png,
    upscale_rgb,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPU geodesic kernel")
    parser.add_argument(
        "--precision",
        choices=["float32", "float64"],
        default="float32",
        help="Kernel internal precision (default: float32)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override resolution from config",
    )
    args = parser.parse_args()

    cfg = load_config()
    data = render_gpu_or_fallback_geodesic(cfg, resolution=args.resolution, precision=args.precision)
    preview = upscale_rgb(data["rgb"])
    suffix = f"_{args.precision}"
    save_png(f"output/gpu_geodesic_image{suffix}.png", preview)
    save_png(f"figures/gpu_geodesic_reference{suffix}.png", preview)
    save_maps_npz(f"reference/gpu_geodesic_reference{suffix}.npz", data)
    meta = metadata_from_render(data)
    meta["backend"] = data.get("backend", "unknown")
    meta["backend_reason"] = data.get("backend_reason", "")
    meta["precision"] = args.precision
    write_json(f"logs/gpu_geodesic_run{suffix}.json", meta)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
