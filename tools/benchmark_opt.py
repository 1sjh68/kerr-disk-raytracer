"""Benchmark float64 baseline vs optimized kernel."""
from __future__ import annotations

import json
import copy
import time

import numpy as np

from src.config import DEFAULT_CONFIG
from src.gpu_trace import render_cuda_geodesic


def benchmark(precision: str, res: int, warmup: bool = True) -> float:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["render"]["geodesic_resolution"] = res
    if warmup:
        render_cuda_geodesic(cfg, resolution=res, precision=precision)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        d = render_cuda_geodesic(cfg, resolution=res, precision=precision)
        t1 = time.perf_counter()
        times.append(d.get("gpu_kernel_elapsed_s", t1 - t0))
    return float(np.median(times))


def main() -> None:
    results = []
    for res in [48, 128, 256]:
        print(f"Benchmarking {res}x{res} ...")
        base = benchmark("float64", res)
        opt = benchmark("float64_opt", res)
        speedup = base / opt if opt > 0 else 0
        results.append({
            "resolution": res,
            "baseline_s": round(base, 4),
            "optimized_s": round(opt, 4),
            "speedup": round(speedup, 2),
        })
        print(f"  baseline={base:.4f}s  optimized={opt:.4f}s  speedup={speedup:.2f}x")

    with open("results/optimization_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    # Validation: check that optimized produces same results as baseline
    print("\nValidating correctness (128x128) ...")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    d_base = render_cuda_geodesic(cfg, resolution=128, precision="float64")
    d_opt = render_cuda_geodesic(cfg, resolution=128, precision="float64_opt")
    status_match = np.mean(d_base["status_code"] == d_opt["status_code"])
    intensity_mae = np.mean(np.abs(d_base["intensity"].astype(float) - d_opt["intensity"].astype(float)))
    print(f"  status_match={status_match*100:.2f}%  intensity_mae={intensity_mae:.6e}")


if __name__ == "__main__":
    main()
