"""Disk geometry and emissivity sweeps (Phase 10 extension).

Two orthogonal axes:
  1. Disk outer radius:    r_outer in {15, 20, 25, 30, 40, 60} (inner = ISCO)
  2. Emissivity index q:   q       in {1.5, 2.0, 2.5, 3.0, 4.0, 5.0}

Fixed: a = 0.7, inclination = 60°, resolution = 128, precision = float64.

Outputs:
  results/disk_radius_sweep.json
  results/disk_emissivity_sweep.json
  figures/disk_radius_comparison.png
  figures/disk_emissivity_comparison.png

Run:
  PYTHONPATH=. .venv/Scripts/python.exe tools/disk_param_sweep.py
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from src.config import DEFAULT_CONFIG
from src.gpu_trace import render_cuda_geodesic
from src.render import save_png, upscale_rgb


RESOLUTION = 128
PRECISION = "float64"
SPIN = 0.7
INCLINATION = 60.0


def _stat_one(data: dict, label: str, varname: str, value: float, image_path: str) -> dict:
    return {
        "label": label,
        "varied": varname,
        "value": value,
        "spin": SPIN,
        "inclination_deg": INCLINATION,
        "resolution": RESOLUTION,
        "precision": PRECISION,
        "kernel_time_s": float(data.get("gpu_kernel_elapsed_s", 0.0)),
        "hit_fraction": float(np.mean(data["hit_mask"] > 0)),
        "intensity_max": float(np.max(data["intensity"])),
        "intensity_mean": float(np.mean(data["intensity"][data["hit_mask"] > 0]) if np.any(data["hit_mask"] > 0) else 0.0),
        "redshift_min": float(np.min(data["redshift"][data["redshift"] > 0]) if np.any(data["redshift"] > 0) else 0.0),
        "redshift_max": float(np.max(data["redshift"])),
        "image_path": image_path,
    }


def _run_one(cfg_overrides: dict, image_name: str) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["black_hole"]["spin"] = SPIN
    cfg["camera"]["inclination_deg"] = INCLINATION
    cfg["render"]["geodesic_resolution"] = RESOLUTION
    for section, kvs in cfg_overrides.items():
        cfg.setdefault(section, {}).update(kvs)

    t0 = time.perf_counter()
    data = render_cuda_geodesic(cfg, resolution=RESOLUTION, precision=PRECISION)
    wall = time.perf_counter() - t0

    preview = upscale_rgb(data["rgb"])
    save_png(image_name, preview)
    print(f"  -> {image_name}  kernel={data.get('gpu_kernel_elapsed_s', 0):.3f}s  wall={wall:.2f}s")
    return data


def sweep_outer_radius() -> list[dict]:
    print("=== disk outer radius sweep ===")
    radii = [15.0, 20.0, 25.0, 30.0, 40.0, 60.0]
    results = []
    for r_out in radii:
        label = f"r_out={r_out:g}"
        path = f"figures/sweep/disk_radius_r_out{int(r_out)}.png"
        data = _run_one(
            cfg_overrides={"disk": {"outer_radius": r_out}},
            image_name=path,
        )
        results.append(_stat_one(data, label, "outer_radius", r_out, path))
    return results


def sweep_emissivity_index() -> list[dict]:
    print("=== emissivity index q sweep ===")
    qs = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = []
    for q in qs:
        label = f"q={q:g}"
        path = f"figures/sweep/disk_emissivity_q{q:g}.png"
        data = _run_one(
            cfg_overrides={"disk": {"emissivity_index": q}},
            image_name=path,
        )
        results.append(_stat_one(data, label, "emissivity_index", q, path))
    return results


def make_comparison_figure(results: list[dict], title: str, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable, skipping figure")
        return
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4 * rows))
    axes = np.atleast_2d(axes).flatten()
    for ax, r in zip(axes, results):
        img = plt.imread(r["image_path"])
        ax.imshow(img)
        ax.set_title(
            f"{r['label']}  hit={r['hit_fraction']:.2f}\n"
            f"I_max={r['intensity_max']:.2g}  g∈[{r['redshift_min']:.2f},{r['redshift_max']:.2f}]",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    Path("figures/sweep").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    radius_results = sweep_outer_radius()
    with open("results/disk_radius_sweep.json", "w", encoding="utf-8") as f:
        json.dump(radius_results, f, indent=2, ensure_ascii=False)
    make_comparison_figure(
        radius_results,
        "Disk outer radius sweep (a=0.7, i=60°, q=3, 128×128 float64)",
        "figures/disk_radius_comparison.png",
    )

    emissivity_results = sweep_emissivity_index()
    with open("results/disk_emissivity_sweep.json", "w", encoding="utf-8") as f:
        json.dump(emissivity_results, f, indent=2, ensure_ascii=False)
    make_comparison_figure(
        emissivity_results,
        "Disk emissivity index q sweep (a=0.7, i=60°, r_out=28, 128×128 float64)",
        "figures/disk_emissivity_comparison.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
