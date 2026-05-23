from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import deep_update, load_config
from src.gpu_trace import cuda_available, render_cuda
from src.render import render_thin_disk_fast, write_json


def main() -> None:
    cfg = load_config()
    cuda_ok, cuda_reason = cuda_available()
    if cuda_ok:
        render_cuda(cfg, resolution=64, block=(16, 16))
    baseline = []
    optimized = []
    block_results = []
    for res in [128, 256, 512, 1024]:
        t0 = time.perf_counter()
        data = render_thin_disk_fast(cfg, resolution=res)
        elapsed = time.perf_counter() - t0
        baseline.append({"resolution": res, "time_s": elapsed, "method": data["method"], "backend": "cpu"})

        if cuda_ok:
            gpu_data = render_cuda(cfg, resolution=res, block=(16, 16))
            optimized.append(
                {
                    "resolution": res,
                    "time_s": gpu_data["elapsed_s"],
                    "gpu_kernel_elapsed_s": gpu_data["gpu_kernel_elapsed_s"],
                    "method": gpu_data["method"],
                    "backend": "cuda",
                    "backend_reason": gpu_data["backend_reason"],
                    "cuda_block": gpu_data["cuda_block"],
                    "speedup_vs_cpu": elapsed / gpu_data["elapsed_s"] if gpu_data["elapsed_s"] > 0 else 0.0,
                }
            )
        else:
            opt_cfg = deep_update(cfg, {"render": {"bloom": False}})
            t1 = time.perf_counter()
            opt_data = render_thin_disk_fast(opt_cfg, resolution=res)
            opt_elapsed = time.perf_counter() - t1
            optimized.append(
                {
                    "resolution": res,
                    "time_s": opt_elapsed,
                    "method": opt_data["method"] + "_no_bloom",
                    "backend": "cpu_fallback",
                    "backend_reason": cuda_reason,
                    "speedup_vs_cpu": elapsed / opt_elapsed if opt_elapsed > 0 else 0.0,
                }
            )

    if cuda_ok:
        for block in [(8, 8), (16, 16), (32, 8)]:
            gpu_data = render_cuda(cfg, resolution=512, block=block)
            block_results.append(
                {
                    "resolution": 512,
                    "cuda_block": f"{block[0]}x{block[1]}",
                    "time_s": gpu_data["elapsed_s"],
                    "gpu_kernel_elapsed_s": gpu_data["gpu_kernel_elapsed_s"],
                    "backend": "cuda",
                }
            )

    write_json("results/performance_baseline.json", {"cuda_available": cuda_ok, "cuda_reason": cuda_reason, "runs": baseline})
    write_json("results/performance_optimized.json", {"cuda_available": cuda_ok, "cuda_reason": cuda_reason, "runs": optimized, "block_results": block_results})

    xs = [row["resolution"] for row in baseline]
    b = [row["time_s"] for row in baseline]
    o = [row["time_s"] for row in optimized]
    speedup = [row["speedup_vs_cpu"] for row in optimized]

    Path("results").mkdir(exist_ok=True)
    with Path("results/parameter_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resolution", "baseline_s", "optimized_s", "speedup", "optimized_backend"])
        writer.writeheader()
        for row, bb, oo, sp in zip(optimized, b, o, speedup):
            writer.writerow({"resolution": row["resolution"], "baseline_s": bb, "optimized_s": oo, "speedup": sp, "optimized_backend": row["backend"]})

    plt.figure(figsize=(6.0, 4.0), dpi=160)
    plt.plot(xs, speedup, marker="o")
    plt.xlabel("resolution")
    plt.ylabel("CPU/GPU speedup" if cuda_ok else "speedup")
    plt.title("CUDA RawKernel speedup vs CPU" if cuda_ok else "MVP render speedup: bloom off vs baseline")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("results/speedup_chart.png")
    plt.close()

    if block_results:
        plt.figure(figsize=(6.0, 4.0), dpi=160)
        labels = [row["cuda_block"] for row in block_results]
        times = [row["gpu_kernel_elapsed_s"] for row in block_results]
        plt.bar(labels, times)
        plt.xlabel("CUDA block")
        plt.ylabel("kernel time (s)")
        plt.title("CUDA block-size benchmark at 512x512")
        plt.tight_layout()
        plt.savefig("results/gpu_block_size.png")
        plt.close()


if __name__ == "__main__":
    main()
