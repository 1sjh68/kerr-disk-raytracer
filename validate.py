from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from src.config import load_config
from src.gpu_trace import render_gpu_or_fallback
from src.render import render_thin_disk_fast, save_png, write_json
from src.safe_io import read_limited_json


def compare_maps(cpu: dict, gpu: dict) -> dict[str, float | str]:
    diff = np.asarray(cpu["intensity"], dtype=float) - np.asarray(gpu["intensity"], dtype=float)
    rgb_diff = np.asarray(cpu["rgb"], dtype=float) - np.asarray(gpu["rgb"], dtype=float)
    denom = max(float(np.mean(np.abs(cpu["intensity"]))), 1.0e-12)
    return {
        "backend": str(gpu.get("backend", "unknown")),
        "backend_reason": str(gpu.get("backend_reason", "")),
        "intensity_mse": float(np.mean(diff * diff)),
        "intensity_mae": float(np.mean(np.abs(diff))),
        "intensity_max_error": float(np.max(np.abs(diff))),
        "intensity_relative_mae": float(np.mean(np.abs(diff)) / denom),
        "hit_mask_mismatch_fraction": float(np.mean(cpu["hit_mask"] != gpu["hit_mask"])),
        "rgb_mse": float(np.mean(rgb_diff * rgb_diff)),
        "rgb_mae": float(np.mean(np.abs(rgb_diff))),
        "rgb_max_error": float(np.max(np.abs(rgb_diff))),
    }


def write_error_summary(path: str | Path, metrics: dict, pytest_returncode: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geodesic_log_path = Path("logs/cpu_geodesic_run.json")
    geodesic = None
    if geodesic_log_path.exists():
        geodesic = read_limited_json(geodesic_log_path)
    lines = [
        "# Validation Error Summary",
        "",
        "## Status",
        "",
        f"- pytest_returncode: `{pytest_returncode}`",
        f"- backend: `{metrics['backend']}`",
        f"- backend_reason: `{metrics['backend_reason']}`",
        "",
        "## CPU/GPU Map Metrics",
        "",
        f"- intensity_mse: `{metrics['intensity_mse']:.6e}`",
        f"- intensity_mae: `{metrics['intensity_mae']:.6e}`",
        f"- intensity_max_error: `{metrics['intensity_max_error']:.6e}`",
        f"- intensity_relative_mae: `{metrics['intensity_relative_mae']:.6e}`",
        f"- hit_mask_mismatch_fraction: `{metrics['hit_mask_mismatch_fraction']:.6e}`",
        f"- rgb_mse: `{metrics['rgb_mse']:.6e}`",
        f"- rgb_mae: `{metrics['rgb_mae']:.6e}`",
        f"- rgb_max_error: `{metrics['rgb_max_error']:.6e}`",
        "",
    ]
    if geodesic is not None:
        lines += [
            "## Hamiltonian Geodesic CPU Reference",
            "",
            f"- method: `{geodesic['method']}`",
            f"- resolution: `{geodesic['resolution']}x{geodesic['resolution']}`",
            f"- elapsed_s: `{geodesic['elapsed_s']:.3f}`",
            f"- hit_fraction: `{geodesic['hit_fraction']:.6f}`",
            f"- status_counts: `{geodesic['status_counts']}`",
            f"- disk_null_error_mean: `{geodesic.get('disk_null_error_mean', 0.0):.6e}`",
            f"- disk_null_error_max: `{geodesic.get('disk_null_error_max', 0.0):.6e}`",
            "",
        ]
    external_path = Path("research/external_demo_summary.json")
    if external_path.exists():
        external = read_limited_json(external_path)
        geokerr = external.get("geokerr", {})
        odyssey = external.get("odyssey", {})
        wsl2_ok = odyssey.get("wsl2_build_succeeded", False)
        lines += [
            "## External Demo Status",
            "",
            f"- geokerr_available: `{geokerr.get('available', False)}`",
            f"- geokerr_parsed_geodesics: `{geokerr.get('parsed_pairs', 0)}`",
            f"- odyssey_docker_build_succeeded: `{odyssey.get('docker_build_succeeded', odyssey.get('build_succeeded', False))}`",
            f"- odyssey_wsl2_build_succeeded: `{wsl2_ok}`",
            f"- odyssey_cuda_header_missing: `{odyssey.get('contains_cuda_header_error', False)}`",
            "",
        ]
    geodesic_cmp_path = Path("validation/geodesic_cpu_gpu_comparison.json")
    if geodesic_cmp_path.exists():
        geodesic_cmp = read_limited_json(geodesic_cmp_path)
        f64 = geodesic_cmp.get("float64", {})
        if f64:
            lines += [
                "## Geodesic CPU/GPU (see validate_geodesic.py)",
                "",
                f"- float64_status_match_fraction: `{f64.get('status_match_fraction', 0.0):.6f}`",
                f"- float64_intensity_mae: `{f64.get('intensity_mae', 0.0):.6e}`",
                f"- float64_redshift_mae: `{f64.get('redshift_mae', 0.0):.6e}`",
                "",
            ]
    if metrics["backend"] == "cuda":
        lines += [
            "## Interpretation",
            "",
            "- This script validates the **fast thin-disk** CPU/GPU output contract only.",
            "- Full per-pixel Hamiltonian geodesic validation is in `validate_geodesic.py` → `validation/geodesic_cpu_gpu_comparison.json`.",
            f"- float64 geodesic kernel @ 48×48 achieves **{f64.get('status_match_fraction', 0.0) * 100:.2f}% status match** with CPU when geodesic comparison JSON is present.",
            "- Use `run_geodesic_gpu.py --precision float64` for scientific reference; float32 is for fast preview.",
        ]
    else:
        lines += [
            "## Interpretation",
            "",
            "- If backend is `cpu_fallback`, this comparison proves fast-path output-format consistency, not CUDA acceleration.",
            "- Geodesic CUDA validation requires a visible CUDA device and `validate_geodesic.py`.",
            "- float64 geodesic @ 48×48 is the authoritative CPU/GPU agreement benchmark when available.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_config()
    cpu = render_thin_disk_fast(cfg)
    gpu = render_gpu_or_fallback(cfg)
    metrics = compare_maps(cpu, gpu)

    diff = np.abs(np.asarray(cpu["intensity"], dtype=float) - np.asarray(gpu["intensity"], dtype=float))
    if np.max(diff) > 0.0:
        diff_image = diff / np.max(diff)
    else:
        diff_image = diff
    save_png("results/error_map.png", np.dstack([diff_image, diff_image, diff_image]))
    comparison = np.concatenate([cpu["rgb"], gpu["rgb"], np.dstack([diff_image] * 3)], axis=1)
    save_png("results/comparison_grid.png", comparison)
    write_json("results/cpu_gpu_comparison.json", metrics)

    result = subprocess.run([sys.executable, "-m", "pytest", "tests"], text=True)
    write_error_summary("validation/error_summary.md", metrics, result.returncode)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
