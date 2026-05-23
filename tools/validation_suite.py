"""Extended validation suite for Kerr geodesic kernel.

Tests:
1. Schwarzschild shadow (a=0) symmetry and radius.
2. Step-size convergence of intensity map.
3. Spin sequence consistency.
4. High/low inclination sanity checks.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from src.config import DEFAULT_CONFIG
from src.gpu_trace import render_cuda_geodesic


def schwarzschild_shadow_test() -> dict:
    """a=0: GPU captured region should match CPU captured region (pixel-wise)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["black_hole"]["spin"] = 0.0
    cfg["camera"]["inclination_deg"] = 60.0
    res = 128

    # GPU
    gpu = render_cuda_geodesic(cfg, resolution=res, precision="float64")
    # CPU (only run once at low res since it's slow)
    from src.render import render_thin_disk_geodesic_cpu
    cpu = render_thin_disk_geodesic_cpu(cfg, resolution=res)

    status_match = cpu["status_code"] == gpu["status_code"]
    match_frac = float(np.mean(status_match))

    result = {
        "test": "schwarzschild_shadow",
        "passed": match_frac >= 0.99,
        "status_match_fraction": round(match_frac, 4),
        "cpu_captured": int(np.sum(cpu["status_code"] == 2)),
        "gpu_captured": int(np.sum(gpu["status_code"] == 2)),
        "cpu_disk": int(np.sum(cpu["status_code"] == 1)),
        "gpu_disk": int(np.sum(gpu["status_code"] == 1)),
    }
    return result


def step_convergence_test() -> dict:
    """Intensity map should converge as step_size decreases."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["black_hole"]["spin"] = 0.7
    cfg["camera"]["inclination_deg"] = 60.0
    res = 128
    step_sizes = [0.5, 0.35, 0.25, 0.15]

    intensities = []
    for h in step_sizes:
        cfg["integration"]["step_size"] = h
        data = render_cuda_geodesic(cfg, resolution=res, precision="float64")
        intensities.append(data["intensity"].astype(float).copy())

    # Compare finest (0.15) against coarser grids
    fine = intensities[-1]
    errors = []
    for coarse in intensities[:-1]:
        diff = np.abs(coarse - fine)
        mask = (coarse > 0) | (fine > 0)
        if np.any(mask):
            errors.append(float(np.mean(diff[mask])))
        else:
            errors.append(0.0)

    result = {
        "test": "step_convergence",
        "passed": all(e < 1.0e-4 for e in errors),
        "step_sizes": step_sizes,
        "mean_abs_errors_vs_finest": [round(e, 6) for e in errors],
    }
    return result


def spin_consistency_test() -> dict:
    """Higher spin should increase hit fraction (larger ISCO, smaller shadow)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["camera"]["inclination_deg"] = 60.0
    res = 128
    spins = [0.0, 0.3, 0.5, 0.7, 0.9, 0.998]

    hit_fracs = []
    for a in spins:
        cfg["black_hole"]["spin"] = a
        data = render_cuda_geodesic(cfg, resolution=res, precision="float64")
        hit_fracs.append(float(np.mean(data["hit_mask"] > 0)))

    # Monotonic increase check (not strict, just trend)
    increases = sum(1 for i in range(1, len(hit_fracs)) if hit_fracs[i] >= hit_fracs[i-1])
    trend_ok = increases >= len(hit_fracs) - 2  # allow one exception

    result = {
        "test": "spin_consistency",
        "passed": trend_ok,
        "spins": spins,
        "hit_fractions": [round(h, 4) for h in hit_fracs],
        "monotonic_increases": increases,
    }
    return result


def inclination_sanity_test() -> dict:
    """Edge-on (i~90) should have lower hit fraction and stronger Doppler asymmetry."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["black_hole"]["spin"] = 0.7
    res = 128

    incs = [10.0, 30.0, 60.0, 80.0]
    hit_fracs = []
    asymmetries = []

    for inc in incs:
        cfg["camera"]["inclination_deg"] = inc
        data = render_cuda_geodesic(cfg, resolution=res, precision="float64")
        hit_fracs.append(float(np.mean(data["hit_mask"] > 0)))

        # Doppler asymmetry: compare left/right redshift in disk region
        red = data["redshift"]
        mask = data["hit_mask"] > 0
        if np.any(mask):
            mid = res // 2
            left = red[:, :mid]
            right = red[:, mid:]
            lmean = np.mean(left[left > 0]) if np.any(left > 0) else 1.0
            rmean = np.mean(right[right > 0]) if np.any(right > 0) else 1.0
            asymmetries.append(float(abs(lmean - rmean) / max(lmean, rmean)))
        else:
            asymmetries.append(0.0)

    # Higher inclination should generally have lower hit fraction
    trend_ok = hit_fracs[-1] < hit_fracs[0]
    # Higher inclination should have stronger asymmetry
    asym_ok = asymmetries[-1] > asymmetries[0]

    result = {
        "test": "inclination_sanity",
        "passed": trend_ok and asym_ok,
        "inclinations": incs,
        "hit_fractions": [round(h, 4) for h in hit_fracs],
        "doppler_asymmetries": [round(a, 4) for a in asymmetries],
    }
    return result


def main() -> None:
    Path("validation").mkdir(exist_ok=True)
    results = []

    tests = [
        schwarzschild_shadow_test,
        step_convergence_test,
        spin_consistency_test,
        inclination_sanity_test,
    ]

    for test_fn in tests:
        print(f"Running {test_fn.__name__} ...")
        r = test_fn()
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {status}: {json.dumps(r, indent=2)}")

    all_passed = all(r["passed"] for r in results)
    summary = {
        "all_passed": all_passed,
        "tests": results,
    }
    with open("validation/extended_validation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Extended Validation Summary ===")
    print(f"All passed: {all_passed}")
    for r in results:
        icon = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"  {icon} {r['test']}")


if __name__ == "__main__":
    main()
