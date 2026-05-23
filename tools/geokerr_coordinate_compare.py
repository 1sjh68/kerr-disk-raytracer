"""Coordinate-level alignment of geokerr per-step trajectories with our CPU tracer.

Strategy
--------
geokerr `abgrid_r60.out` records 300 sample points per ray as
(uf, muf, dt, dphi, lambda, tpm, tpr). The affine parameter `lambda` does
not match our convention bit-for-bit, so we align in the (r, θ) plane
geometrically: for each geokerr point we find the nearest point on our
trajectory by Euclidean distance, then report per-ray statistics on the
positional error.

We pick 5 rays covering different fates (disk hit, captured, near-shadow
boundary, off-axis, far) and produce both a JSON summary and an overlay
plot.

Outputs
-------
validation/geokerr_coordinate_alignment.json
research/reproduction/geokerr_coordinate_alignment.png
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from scripts.parse_geokerr import parse_abgrid_out
from src.geodesic import trace_single_ray


# 5 representative (alpha, beta) picks from abgrid_r60.out's 20x20 grid
# Coordinates are reused from observed geokerr ray catalog.
SAMPLE_RAYS = [
    (0.6, 0.6),       # near optical axis (high deflection)
    (-3.0, -3.0),     # quadrant I disk hit
    (5.4, -5.4),      # large impact parameter, escapes
    (3.0, 3.0),       # opposite-side disk hit
    (-1.8, -1.8),     # near critical / captured
]


def nearest_neighbour_error(
    geokerr_pts: np.ndarray, our_traj: np.ndarray
) -> dict:
    """For each geokerr point, find the L2-nearest point on our trajectory.

    Both arrays are normalised by characteristic scales to make r and theta
    contributions comparable.
    """
    if our_traj is None or len(our_traj) < 2:
        return {"matched": 0, "rms_dr_over_r": float("nan"), "rms_dtheta_rad": float("nan")}

    g_r = geokerr_pts[:, 0]
    g_theta = geokerr_pts[:, 1]
    o_r = our_traj[:, 1]
    o_theta = our_traj[:, 2]

    # Normalise distance: dr/r ~ relative; dtheta in radians
    # Search nearest using a (r, theta) tuple distance: log(r) + theta
    g_logr = np.log(np.maximum(g_r, 1.0e-6))
    o_logr = np.log(np.maximum(o_r, 1.0e-6))

    dr_rel_list = []
    dtheta_list = []
    for i in range(len(g_r)):
        d = (o_logr - g_logr[i]) ** 2 + (o_theta - g_theta[i]) ** 2
        j = int(np.argmin(d))
        dr_rel_list.append(abs(o_r[j] - g_r[i]) / max(g_r[i], 1.0e-6))
        dtheta_list.append(abs(o_theta[j] - g_theta[i]))

    dr_arr = np.asarray(dr_rel_list)
    dtheta_arr = np.asarray(dtheta_list)
    return {
        "matched": int(len(g_r)),
        "rms_dr_over_r": float(np.sqrt(np.mean(dr_arr**2))),
        "p95_dr_over_r": float(np.percentile(dr_arr, 95)),
        "max_dr_over_r": float(np.max(dr_arr)),
        "rms_dtheta_rad": float(np.sqrt(np.mean(dtheta_arr**2))),
        "p95_dtheta_rad": float(np.percentile(dtheta_arr, 95)),
        "max_dtheta_rad": float(np.max(dtheta_arr)),
    }


def main() -> None:
    geokerr_path = Path("research/repos/geokerr/geokerr_code/abgrid_r60.out")
    print(f"Parsing {geokerr_path}")
    data = parse_abgrid_out(geokerr_path)
    a = data["a"]
    inclination_deg = data["inclination_deg"]
    r_obs_geokerr = data["r_obs"]
    print(f"  a={a}, i={inclination_deg:.1f}°, r_obs_geokerr={r_obs_geokerr:.2e}")

    # Index by (alpha, beta) for fast lookup
    by_ab = {}
    for ray in data["rays"]:
        key = (round(float(ray["alpha"]), 2), round(float(ray["beta"]), 2))
        by_ab[key] = ray

    out_records = []
    plot_data = []  # for the overlay figure

    # Practical r_obs for our tracer (1000 ~ "infinity" in M units)
    r_obs_cpu = 1000.0
    max_steps = 5000
    step_size = 0.35

    for alpha, beta in SAMPLE_RAYS:
        key = (round(alpha, 2), round(beta, 2))
        if key not in by_ab:
            print(f"  alpha={alpha}, beta={beta} not in geokerr grid, skipping")
            continue
        ray = by_ab[key]
        print(f"\n=== ray (alpha={alpha:+.1f}, beta={beta:+.1f}, ncase={ray['ncase']}, nup={ray['nup']}) ===")

        # geokerr per-step (r, theta)
        gpts = np.array(
            [(p["r"], p["theta"]) for p in ray["points"]], dtype=float
        )

        # Our CPU trace with trajectory recording
        t0 = time.perf_counter()
        hit = trace_single_ray(
            alpha, beta,
            a=a, inclination_deg=inclination_deg,
            r_obs=r_obs_cpu,
            step_size=step_size, max_steps=max_steps,
            escape_radius=90.0,
            record_trajectory=True,
        )
        wall = time.perf_counter() - t0
        traj = hit.trajectory
        print(f"  CPU: status={hit.status} steps={hit.steps} traj_len={0 if traj is None else len(traj)} wall={wall:.2f}s")

        err = nearest_neighbour_error(gpts, traj)
        print(f"  alignment: rms dr/r={err.get('rms_dr_over_r', 'nan'):.3e} "
              f"rms dtheta={err.get('rms_dtheta_rad', 'nan'):.3e} rad")

        out_records.append({
            "alpha": alpha,
            "beta": beta,
            "ncase": ray["ncase"],
            "geokerr_nup": ray["nup"],
            "cpu_status": hit.status,
            "cpu_steps": hit.steps,
            "alignment_error": err,
        })
        plot_data.append((alpha, beta, gpts, traj, hit.status))

    # Overall stats
    overall = {
        "rays_compared": len(out_records),
        "median_rms_dr_over_r": float(np.median([r["alignment_error"]["rms_dr_over_r"]
                                                  for r in out_records
                                                  if math.isfinite(r["alignment_error"].get("rms_dr_over_r", float("nan")))])),
        "median_rms_dtheta_rad": float(np.median([r["alignment_error"]["rms_dtheta_rad"]
                                                   for r in out_records
                                                   if math.isfinite(r["alignment_error"].get("rms_dtheta_rad", float("nan")))])),
    }
    print(f"\n=== overall: {overall}")

    Path("validation").mkdir(exist_ok=True)
    out_json = {
        "geokerr_path": str(geokerr_path),
        "a": a,
        "inclination_deg": inclination_deg,
        "r_obs_geokerr": r_obs_geokerr,
        "r_obs_cpu": r_obs_cpu,
        "step_size": step_size,
        "max_steps": max_steps,
        "rays": out_records,
        "overall": overall,
    }
    with open("validation/geokerr_coordinate_alignment.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print("Saved validation/geokerr_coordinate_alignment.json")

    # Overlay plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable, skipping figure")
        return
    Path("research/reproduction").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    for ax, (alpha, beta, gpts, traj, status) in zip(axes, plot_data):
        ax.semilogx(gpts[:, 0], gpts[:, 1], "r-", label="geokerr", lw=1.0, alpha=0.8)
        if traj is not None and len(traj) >= 2:
            ax.semilogx(traj[:, 1], traj[:, 2], "b--", label="our CPU", lw=0.8, alpha=0.8)
        ax.axhline(np.pi / 2.0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("r [M]")
        ax.set_ylabel(r"$\theta$ [rad]")
        ax.set_title(f"({alpha:+.1f}, {beta:+.1f}) {status}", fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[len(plot_data):]:
        ax.axis("off")
    fig.suptitle(f"Coordinate-level alignment: geokerr vs CPU tracer (a={a}, i={inclination_deg:.0f}°)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("research/reproduction/geokerr_coordinate_alignment.png", dpi=150)
    plt.close()
    print("Saved research/reproduction/geokerr_coordinate_alignment.png")


if __name__ == "__main__":
    main()
