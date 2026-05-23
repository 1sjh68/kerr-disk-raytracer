from __future__ import annotations

import json
import math

import numpy as np

from src.config import load_config
from src.gpu_trace import render_cuda_geodesic
from src.render import render_thin_disk_geodesic_cpu, write_json


def compare_cpu_gpu(cpu: dict, gpu: dict, label: str, cfg: dict) -> dict:
    mask = (cpu["hit_mask"] > 0) | (gpu["hit_mask"] > 0)
    status_match = cpu["status_code"] == gpu["status_code"]

    results: dict = {
        "label": label,
        "resolution": int(cpu.get("resolution", cfg["render"].get("geodesic_resolution", 48))),
        "spin": float(cfg["black_hole"]["spin"]),
        "inclination_deg": float(cfg["camera"]["inclination_deg"]),
        "status_match_fraction": float(np.mean(status_match)),
        "cpu_backend": "cpu",
        "gpu_backend": gpu.get("backend", "unknown"),
    }

    if np.any(mask):
        for key in ["intensity", "redshift", "temperature"]:
            diff = np.abs(cpu[key].astype(float) - gpu[key].astype(float))
            results[f"{key}_mse"] = float(np.mean(diff[mask] ** 2))
            results[f"{key}_mae"] = float(np.mean(diff[mask]))
            results[f"{key}_max_error"] = float(np.max(diff[mask]))
    else:
        for key in ["intensity", "redshift", "temperature"]:
            results[f"{key}_mse"] = 0.0
            results[f"{key}_mae"] = 0.0
            results[f"{key}_max_error"] = 0.0

    results["cpu_null_error_mean"] = float(np.mean(cpu["null_error"]))
    results["gpu_null_error_mean"] = float(np.mean(gpu["null_error"]))
    results["cpu_null_error_max"] = float(np.max(cpu["null_error"]))
    results["gpu_null_error_max"] = float(np.max(gpu["null_error"]))

    for name in ["disk", "captured", "escaped", "max_steps", "invalid"]:
        code = {"disk": 1, "captured": 2, "escaped": 3, "max_steps": 4, "invalid": 5}[name]
        results[f"cpu_{name}"] = int(np.sum(cpu["status_code"] == code))
        results[f"gpu_{name}"] = int(np.sum(gpu["status_code"] == code))

    return results


def main() -> None:
    cfg = load_config()
    res = int(cfg["render"].get("geodesic_resolution", 48))

    print(f"Running CPU geodesic ({res}x{res}) ...")
    cpu = render_thin_disk_geodesic_cpu(cfg, resolution=res)
    print(f"  elapsed: {cpu['elapsed_s']:.3f}s")

    all_results = {}

    for precision in ("float32", "float64"):
        print(f"Running GPU geodesic ({res}x{res}, {precision}) ...")
        try:
            gpu = render_cuda_geodesic(cfg, resolution=res, precision=precision)
            print(f"  elapsed: {gpu['elapsed_s']:.3f}s  kernel: {gpu.get('gpu_kernel_elapsed_s', 0):.4f}s")
            r = compare_cpu_gpu(cpu, gpu, precision, cfg)
            all_results[precision] = r
            print(json.dumps(r, indent=2))
            print(f"\n--- {precision.upper()} Summary ---")
            print(f"Status match: {r['status_match_fraction']*100:.2f}%")
            print(f"Intensity MAE: {r['intensity_mae']:.6e}")
            print(f"Redshift MAE: {r['redshift_mae']:.6e}")
            print(f"CPU null error mean: {r['cpu_null_error_mean']:.6e}")
            print(f"GPU null error mean: {r['gpu_null_error_mean']:.6e}")
            print(f"CPU captured/escaped: {r['cpu_captured']}/{r['cpu_escaped']}")
            print(f"GPU captured/escaped: {r['gpu_captured']}/{r['gpu_escaped']}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            all_results[precision] = {"error": str(exc)}

    write_json("validation/geodesic_cpu_gpu_comparison.json", all_results)

    # Final delta summary
    if "float32" in all_results and "float64" in all_results and "error" not in all_results["float32"] and "error" not in all_results["float64"]:
        f32 = all_results["float32"]
        f64 = all_results["float64"]
        print("\n=== PRECISION COMPARISON ===")
        print(f"float32 status match: {f32['status_match_fraction']*100:.2f}%")
        print(f"float64 status match: {f64['status_match_fraction']*100:.2f}%")
        print(f"Improvement: {(f64['status_match_fraction'] - f32['status_match_fraction'])*100:.2f}pp")


if __name__ == "__main__":
    main()
