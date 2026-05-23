"""RK45 (Dormand-Prince) vs RK4 fixed-step on the same critical rays.

This is a direct follow-up to validation/geokerr_coordinate_alignment.md
which identified RK4 fixed-step degradation near the photon sphere as
the main source of trajectory-level disagreement with geokerr's
Carlson-elliptic semi-analytic integration.

For each of the 5 representative rays we:
  1. Run RK4 with the project default (step=0.35, max_steps=5000)
  2. Run RK45 (atol=1e-8, rtol=1e-6, max_steps=5000)
  3. Compare: status, accepted steps, null constraint error,
     geometric (r, theta) RMS distance to geokerr ground truth.

Output: validation/rk45_vs_rk4_demo.json + figure
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
from src.geodesic import trace_single_ray, trace_single_ray_rk45


SAMPLE_RAYS = [
    (0.6, 0.6),
    (-3.0, -3.0),
    (5.4, -5.4),
    (3.0, 3.0),
    (-1.8, -1.8),
]


def nearest_neighbour_rms(geokerr_pts: np.ndarray, traj: np.ndarray | None) -> float:
    if traj is None or len(traj) < 2:
        return float("nan")
    g_logr = np.log(np.maximum(geokerr_pts[:, 0], 1.0e-6))
    g_theta = geokerr_pts[:, 1]
    o_logr = np.log(np.maximum(traj[:, 1], 1.0e-6))
    o_theta = traj[:, 2]
    dr_rel = []
    dth = []
    for i in range(len(g_logr)):
        d2 = (o_logr - g_logr[i]) ** 2 + (o_theta - g_theta[i]) ** 2
        j = int(np.argmin(d2))
        dr_rel.append(abs(np.exp(o_logr[j]) - np.exp(g_logr[i])) / max(np.exp(g_logr[i]), 1e-6))
        dth.append(abs(o_theta[j] - g_theta[i]))
    return {
        "rms_dr_over_r": float(np.sqrt(np.mean(np.array(dr_rel) ** 2))),
        "rms_dtheta_rad": float(np.sqrt(np.mean(np.array(dth) ** 2))),
    }


def main() -> None:
    geokerr_path = Path("research/repos/geokerr/geokerr_code/abgrid_r60.out")
    print(f"Parsing {geokerr_path}")
    data = parse_abgrid_out(geokerr_path)
    a = data["a"]
    inclination_deg = data["inclination_deg"]
    r_obs_cpu = 1000.0

    by_ab = {}
    for ray in data["rays"]:
        key = (round(float(ray["alpha"]), 2), round(float(ray["beta"]), 2))
        by_ab[key] = ray

    records = []
    plot_data = []

    for alpha, beta in SAMPLE_RAYS:
        key = (round(alpha, 2), round(beta, 2))
        if key not in by_ab:
            print(f"  ({alpha},{beta}) not in geokerr, skip")
            continue
        gpts = np.array([(p["r"], p["theta"]) for p in by_ab[key]["points"]], dtype=float)
        gstatus_ncase = by_ab[key]["ncase"]

        print(f"\n=== ray (alpha={alpha:+.1f}, beta={beta:+.1f}, geokerr ncase={gstatus_ncase}) ===")

        t0 = time.perf_counter()
        rk4 = trace_single_ray(
            alpha, beta, a=a, inclination_deg=inclination_deg,
            r_obs=r_obs_cpu, max_steps=5000, record_trajectory=True,
        )
        t_rk4 = time.perf_counter() - t0

        t0 = time.perf_counter()
        rk45 = trace_single_ray_rk45(
            alpha, beta, a=a, inclination_deg=inclination_deg,
            r_obs=r_obs_cpu, max_steps=5000, atol=1e-8, rtol=1e-6,
            record_trajectory=True,
        )
        t_rk45 = time.perf_counter() - t0

        rms4 = nearest_neighbour_rms(gpts, rk4.trajectory)
        rms45 = nearest_neighbour_rms(gpts, rk45.trajectory)

        rec = {
            "alpha": alpha, "beta": beta, "geokerr_ncase": gstatus_ncase,
            "rk4": {
                "status": rk4.status, "steps": rk4.steps, "wall_s": round(t_rk4, 4),
                "null_error": float(min(rk4.null_error, 1e30)),
                "rms_dr_over_r": rms4.get("rms_dr_over_r"),
                "rms_dtheta_rad": rms4.get("rms_dtheta_rad"),
            },
            "rk45": {
                "status": rk45.status, "steps": rk45.steps, "wall_s": round(t_rk45, 4),
                "null_error": float(min(rk45.null_error, 1e30)),
                "rms_dr_over_r": rms45.get("rms_dr_over_r"),
                "rms_dtheta_rad": rms45.get("rms_dtheta_rad"),
            },
        }
        records.append(rec)
        plot_data.append((alpha, beta, gpts, rk4.trajectory, rk45.trajectory,
                          rk4.status, rk45.status))
        print(f"  RK4  : status={rk4.status:8s} steps={rk4.steps:5d} wall={t_rk4:.3f}s "
              f"null={rec['rk4']['null_error']:.3e} rms_dr/r={rms4.get('rms_dr_over_r', float('nan')):.3e}")
        print(f"  RK45 : status={rk45.status:8s} steps={rk45.steps:5d} wall={t_rk45:.3f}s "
              f"null={rec['rk45']['null_error']:.3e} rms_dr/r={rms45.get('rms_dr_over_r', float('nan')):.3e}")

    Path("validation").mkdir(exist_ok=True)
    out = {
        "geokerr_path": str(geokerr_path),
        "a": a, "inclination_deg": inclination_deg, "r_obs_cpu": r_obs_cpu,
        "rk4_step_size": 0.35,
        "rk45_atol": 1e-8, "rk45_rtol": 1e-6,
        "rays": records,
    }
    with open("validation/rk45_vs_rk4_demo.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved validation/rk45_vs_rk4_demo.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    Path("research/reproduction").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    for ax, (alpha, beta, gpts, traj4, traj45, st4, st45) in zip(axes, plot_data):
        ax.semilogx(gpts[:, 0], gpts[:, 1], "r-", label="geokerr", lw=1.0, alpha=0.85)
        if traj4 is not None and len(traj4) >= 2:
            ax.semilogx(traj4[:, 1], traj4[:, 2], "b--", label=f"RK4 ({st4})", lw=0.7, alpha=0.7)
        if traj45 is not None and len(traj45) >= 2:
            ax.semilogx(traj45[:, 1], traj45[:, 2], "g-.", label=f"RK45 ({st45})", lw=0.9, alpha=0.85)
        ax.axhline(np.pi / 2.0, color="gray", lw=0.4, ls=":")
        ax.set_xlabel("r [M]")
        ax.set_ylabel(r"$\theta$ [rad]")
        ax.set_title(f"({alpha:+.1f}, {beta:+.1f})", fontsize=10)
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[len(plot_data):]:
        ax.axis("off")
    fig.suptitle("RK4 vs Dormand-Prince RK45 vs geokerr (a=0.7, i=60°)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("research/reproduction/rk45_vs_rk4.png", dpi=150)
    plt.close()
    print("Saved research/reproduction/rk45_vs_rk4.png")


if __name__ == "__main__":
    main()
