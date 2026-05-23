"""Resolution scaling benchmark for float64 geodesic kernel."""
from __future__ import annotations

import json
import time

import numpy as np

from src.config import load_config
from src.gpu_trace import render_cuda_geodesic


def main() -> None:
    cfg = load_config()
    resolutions = [48, 64, 96, 128, 160, 192, 256]
    results = []

    for res in resolutions:
        # Warm-up
        render_cuda_geodesic(cfg, resolution=res, precision="float64")

        times = []
        kernel_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            data = render_cuda_geodesic(cfg, resolution=res, precision="float64")
            t1 = time.perf_counter()
            times.append(t1 - t0)
            kernel_times.append(data.get("gpu_kernel_elapsed_s", 0))

        avg_wall = float(np.mean(times))
        avg_kernel = float(np.mean(kernel_times))
        pixels = res * res

        results.append({
            "resolution": res,
            "pixels": pixels,
            "wall_time_s": round(avg_wall, 4),
            "kernel_time_s": round(avg_kernel, 4),
            "ms_per_megapixel": round(avg_kernel / (pixels / 1e6), 2),
        })
        print(f"{res}x{res}: kernel={avg_kernel:.4f}s, wall={avg_wall:.4f}s")

    with open("results/resolution_sweep_float64.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Scaling Summary ===")
    base = results[0]
    for r in results[1:]:
        ideal = base["kernel_time_s"] * (r["pixels"] / base["pixels"])
        actual = r["kernel_time_s"]
        efficiency = ideal / actual if actual > 0 else 0
        print(f"{r['resolution']}x{r['resolution']}: {r['ms_per_megapixel']} ms/Mpx  "
              f"(scaling efficiency vs linear: {efficiency*100:.1f}%)")


if __name__ == "__main__":
    main()
