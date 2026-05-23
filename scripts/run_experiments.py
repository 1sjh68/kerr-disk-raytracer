from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import deep_update, load_config
from src.gpu_trace import cuda_available, render_gpu_or_fallback
from src.render import render_thin_disk_fast, save_png


def _grid_image(images: list[np.ndarray], labels: list[str], path: str) -> None:
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(3.0 * cols, 3.0), dpi=160)
    if cols == 1:
        axes = [axes]
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    cuda_ok, cuda_reason = cuda_available()
    rows = []

    spin_images = []
    spin_labels = []
    for spin in [0.0, 0.3, 0.5, 0.7, 0.9, 0.998]:
        run_cfg = deep_update(cfg, {"black_hole": {"spin": spin}, "render": {"resolution": 96}})
        data = render_thin_disk_fast(run_cfg, resolution=96)
        spin_images.append(data["rgb"])
        spin_labels.append(f"a={spin:g}")
        rows.append({
            "case": "spin",
            "spin": spin,
            "inclination_deg": run_cfg["camera"]["inclination_deg"],
            "resolution": 96,
            "elapsed_s": data["elapsed_s"],
            "hit_fraction": float(np.mean(data["hit_mask"] > 0)),
            "intensity_max": float(np.max(data["intensity"])),
        })
    _grid_image(spin_images, spin_labels, "results/spin_comparison.png")

    inc_images = []
    inc_labels = []
    for inc in [10, 30, 60, 80]:
        run_cfg = deep_update(cfg, {"camera": {"inclination_deg": inc}, "render": {"resolution": 128}})
        data = render_thin_disk_fast(run_cfg, resolution=128)
        inc_images.append(data["rgb"])
        inc_labels.append(f"i={inc} deg")
        rows.append({
            "case": "inclination",
            "spin": run_cfg["black_hole"]["spin"],
            "inclination_deg": inc,
            "resolution": 128,
            "elapsed_s": data["elapsed_s"],
            "hit_fraction": float(np.mean(data["hit_mask"] > 0)),
            "intensity_max": float(np.max(data["intensity"])),
        })
    _grid_image(inc_images, inc_labels, "results/inclination_comparison.png")

    res_rows = []
    for res in [256, 512, 1024, 1920]:
        data = render_thin_disk_fast(cfg, resolution=res)
        gpu = render_gpu_or_fallback(cfg, resolution=res)
        speedup = data["elapsed_s"] / gpu["elapsed_s"] if gpu["elapsed_s"] > 0 else 0.0
        res_rows.append((res, data["elapsed_s"], gpu["elapsed_s"], speedup, gpu["backend"]))
        rows.append({
            "case": "resolution",
            "spin": cfg["black_hole"]["spin"],
            "inclination_deg": cfg["camera"]["inclination_deg"],
            "resolution": res,
            "elapsed_s": data["elapsed_s"],
            "gpu_elapsed_s": gpu["elapsed_s"],
            "gpu_backend": gpu["backend"],
            "cpu_gpu_speedup": speedup,
            "hit_fraction": float(np.mean(data["hit_mask"] > 0)),
            "intensity_max": float(np.max(data["intensity"])),
        })

    base_res, base_time, _, _, _ = res_rows[0]
    ideal = [(res / base_res) ** 2 for res, *_ in res_rows]
    throughput = [base_time / t * (res / base_res) ** 2 if t > 0 else 0.0 for res, t, *_ in res_rows]
    cpu_gpu_speedup = [sp for _, _, _, sp, _ in res_rows]
    plt.figure(figsize=(6.2, 4.0), dpi=160)
    plt.plot([r for r, *_ in res_rows], ideal, marker="o", label="pixel-count scaling")
    plt.plot([r for r, *_ in res_rows], throughput, marker="s", label="CPU relative throughput")
    plt.plot([r for r, *_ in res_rows], cpu_gpu_speedup, marker="^", label=f"CPU/{'GPU' if cuda_ok else 'fallback'} speedup")
    plt.xlabel("resolution")
    plt.ylabel("relative factor")
    plt.title("Resolution sweep")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/resolution_speedup.png")
    plt.close()

    with Path("results/parameter_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "spin",
                "inclination_deg",
                "resolution",
                "elapsed_s",
                "gpu_elapsed_s",
                "gpu_backend",
                "cpu_gpu_speedup",
                "hit_fraction",
                "intensity_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    save_png("figures/final_render.png", render_thin_disk_fast(cfg, resolution=256)["rgb"])


if __name__ == "__main__":
    main()
