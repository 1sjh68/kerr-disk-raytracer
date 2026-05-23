"""Benchmark --use_fast_math vs baseline for the float32 geodesic kernel.

For each resolution we:
  1. Run baseline 3 warmup + 5 measure.
  2. Run fastmath 3 warmup + 5 measure.
  3. Compare kernel time (median of 5), accuracy metrics
     (intensity MAE, status / hit_mask match rate, null-error mean).

Output: results/fastmath_benchmark.json + figures/fastmath_speedup.png
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from statistics import median

import numpy as np

from src.config import DEFAULT_CONFIG
from src.gpu_trace import render_cuda_geodesic


RESOLUTIONS = [48, 96, 128, 192, 256, 384, 512]
WARMUP = 3
MEASURE = 5


def time_run(cfg: dict, resolution: int, fast_math: bool) -> tuple[float, dict]:
    times = []
    out = None
    for i in range(WARMUP + MEASURE):
        out = render_cuda_geodesic(
            cfg, resolution=resolution, precision="float32", fast_math=fast_math
        )
        if i >= WARMUP:
            times.append(float(out["gpu_kernel_elapsed_s"]))
    return median(times), out


def main() -> None:
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    records = []

    for res in RESOLUTIONS:
        print(f"--- resolution {res}x{res} ---")
        t_base, base = time_run(cfg, res, fast_math=False)
        t_fast, fast = time_run(cfg, res, fast_math=True)
        speedup = t_base / max(t_fast, 1e-12)

        intensity_mae = float(np.mean(np.abs(base["intensity"] - fast["intensity"])))
        intensity_max = float(np.max(np.abs(base["intensity"] - fast["intensity"])))
        status_match = float(np.mean(base["status_code"] == fast["status_code"]))
        hit_match = float(np.mean(base["hit_mask"] == fast["hit_mask"]))
        null_base = float(np.mean(base["null_error"][base["null_error"] < 1e6]))
        null_fast = float(np.mean(fast["null_error"][fast["null_error"] < 1e6]))

        rec = {
            "resolution": res,
            "baseline_kernel_ms": round(t_base * 1000.0, 4),
            "fastmath_kernel_ms": round(t_fast * 1000.0, 4),
            "speedup": round(speedup, 2),
            "intensity_mae": intensity_mae,
            "intensity_max_abs_diff": intensity_max,
            "status_match_rate": status_match,
            "hit_mask_match_rate": hit_match,
            "null_error_mean_baseline": null_base,
            "null_error_mean_fastmath": null_fast,
        }
        records.append(rec)
        print(
            f"  baseline {t_base*1000:.3f} ms | fastmath {t_fast*1000:.3f} ms | "
            f"speedup {speedup:.2f}x | status {status_match*100:.2f}% | "
            f"intensity MAE {intensity_mae:.2e}"
        )

    out_path = Path("results/fastmath_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "kernel": "kerr_geodesic_kernel (float32)",
                "warmup_runs": WARMUP,
                "measure_runs": MEASURE,
                "summary_metric": "median",
                "runs": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved {out_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable, skipping figure")
        return
    res_arr = [r["resolution"] for r in records]
    base_ms = [r["baseline_kernel_ms"] for r in records]
    fast_ms = [r["fastmath_kernel_ms"] for r in records]
    speedup = [r["speedup"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1 = axes[0]
    ax1.plot(res_arr, base_ms, "o-", label="baseline (sinf/cosf/sqrtf)")
    ax1.plot(res_arr, fast_ms, "s-", label="--use_fast_math (intrinsics)")
    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("Kernel time [ms]")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_title("float32 geodesic kernel time")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(res_arr, speedup, "ro-", lw=2)
    ax2.axhline(1.0, color="gray", ls=":", alpha=0.5)
    ax2.set_xlabel("Resolution")
    ax2.set_ylabel("Speedup (baseline / fastmath)")
    ax2.set_xscale("log", base=2)
    ax2.set_title("--use_fast_math speedup")
    ax2.grid(alpha=0.3)

    fig.suptitle("CUDA fast math intrinsics benchmark (RTX 4060 Laptop)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/fastmath_speedup.png", dpi=150)
    plt.close()
    print("Saved figures/fastmath_speedup.png")


if __name__ == "__main__":
    main()
