"""Full parameter sweep: spin × inclination × resolution (float64 GPU)."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from src.config import DEFAULT_CONFIG
from src.gpu_trace import render_cuda_geodesic
from src.render import save_png, upscale_rgb


def run_sweep() -> list[dict]:
    spins = [0.0, 0.3, 0.5, 0.7, 0.9, 0.998]
    inclinations = [10.0, 30.0, 60.0, 80.0]
    resolution = 128
    precision = "float64"

    results = []
    total = len(spins) * len(inclinations)
    count = 0

    for a in spins:
        for inc in inclinations:
            count += 1
            cfg = copy.deepcopy(DEFAULT_CONFIG)
            cfg["black_hole"]["spin"] = a
            cfg["camera"]["inclination_deg"] = inc
            cfg["render"]["geodesic_resolution"] = resolution

            t0 = time.perf_counter()
            data = render_cuda_geodesic(cfg, resolution=resolution, precision=precision)
            wall = time.perf_counter() - t0

            preview = upscale_rgb(data["rgb"])
            fname = f"figures/sweep_spin{a}_inc{int(inc)}.png"
            save_png(fname, preview)

            meta = {
                "spin": a,
                "inclination_deg": inc,
                "resolution": resolution,
                "precision": precision,
                "kernel_time_s": data.get("gpu_kernel_elapsed_s", 0.0),
                "wall_time_s": round(wall, 4),
                "hit_fraction": float(np.mean(data["hit_mask"] > 0)),
                "intensity_max": float(np.max(data["intensity"])),
                "redshift_min": float(np.min(data["redshift"][data["redshift"] > 0]) if np.any(data["redshift"] > 0) else 0.0),
                "redshift_max": float(np.max(data["redshift"])),
                "status_counts": {
                    "disk": int(np.sum(data["status_code"] == 1)),
                    "captured": int(np.sum(data["status_code"] == 2)),
                    "escaped": int(np.sum(data["status_code"] == 3)),
                    "max_steps": int(np.sum(data["status_code"] == 4)),
                    "invalid": int(np.sum(data["status_code"] == 5)),
                },
                "image_path": fname,
            }
            results.append(meta)
            print(f"[{count}/{total}] a={a:+.3f} i={inc:02.0f}°  kernel={meta['kernel_time_s']:.3f}s  hit={meta['hit_fraction']:.3f}")

    return results


def make_comparison_figures(results: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping figure generation")
        return

    # Spin comparison (fixed inclination 60°)
    spin_results = [r for r in results if abs(r["inclination_deg"] - 60.0) < 1e-6]
    spin_results.sort(key=lambda x: x["spin"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for idx, r in enumerate(spin_results):
        ax = axes[idx]
        img = plt.imread(r["image_path"])
        ax.imshow(img)
        ax.set_title(f"a = {r['spin']:.3f}\nkernel {r['kernel_time_s']:.3f}s", fontsize=10)
        ax.axis("off")
    fig.suptitle("Spin Comparison (i = 60°, 128×128, float64)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/spin_comparison.png", dpi=150)
    plt.close()
    print("Saved results/spin_comparison.png")

    # Inclination comparison (fixed spin 0.7)
    inc_results = [r for r in results if abs(r["spin"] - 0.7) < 1e-6]
    inc_results.sort(key=lambda x: x["inclination_deg"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for idx, r in enumerate(inc_results):
        ax = axes[idx]
        img = plt.imread(r["image_path"])
        ax.imshow(img)
        ax.set_title(f"i = {r['inclination_deg']:.0f}°\nkernel {r['kernel_time_s']:.3f}s", fontsize=10)
        ax.axis("off")
    fig.suptitle("Inclination Comparison (a = 0.7, 128×128, float64)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/inclination_comparison.png", dpi=150)
    plt.close()
    print("Saved results/inclination_comparison.png")


def main() -> None:
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    results = run_sweep()
    with open("results/parameter_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    make_comparison_figures(results)
    print(f"\nDone. Total configurations: {len(results)}")


if __name__ == "__main__":
    main()
