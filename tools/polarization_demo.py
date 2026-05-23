"""Polarization demo: trace disk pixels, build Stokes Q,U maps, plot EVPA.

This is a per-pixel Python demo (CPU only) that runs at low resolution
(default 64×64) for visual inspection. It does NOT replace the GPU
geodesic kernel; it is purely a pedagogical end-to-end test of the
polarization pipeline in src/polarization.py.

For each pixel that hits the disk we:
  1. Trace the ray to its disk-hit point (CPU geodesic).
  2. Build emitted polarization 4-vector f assuming toroidal B field.
  3. Compute the Walker-Penrose constant κ_WP at emission.
  4. Decode observer EVPA from κ_WP and screen position.
  5. Combine with disk intensity to write Stokes I, Q, U maps.

Outputs:
  results/polarization_demo.json
  figures/polarization_stokes_qu.png
  figures/polarization_evpa_quiver.png
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np

from src.camera import screen_grid
from src.config import DEFAULT_CONFIG
from src.geodesic import trace_single_ray
from src.polarization import (
    DEFAULT_PI_LINEAR,
    decode_observer_evpa,
    emitted_polarization_vector_toroidal,
    rotate_stokes_to_observer,
    walker_penrose_complex,
)


RESOLUTION = 48  # CPU geodesic per-pixel is slow; keep modest
PI_LINEAR = DEFAULT_PI_LINEAR


def main() -> None:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    a = float(cfg["black_hole"]["spin"])
    inclination_deg = float(cfg["camera"]["inclination_deg"])
    inclination_rad = math.radians(inclination_deg)
    r_obs = float(cfg["camera"]["r_obs"])
    fov_m = float(cfg["camera"]["fov_m"])

    alpha_grid, beta_grid = screen_grid(RESOLUTION, fov_m)
    H, W = alpha_grid.shape

    I_map = np.zeros((H, W), dtype=float)
    Q_map = np.zeros((H, W), dtype=float)
    U_map = np.zeros((H, W), dtype=float)
    chi_em_map = np.zeros((H, W), dtype=float)
    chi_obs_map = np.zeros((H, W), dtype=float)
    hit_mask = np.zeros((H, W), dtype=bool)
    kappa_re_map = np.zeros((H, W), dtype=float)
    kappa_im_map = np.zeros((H, W), dtype=float)

    print(f"=== Polarization demo {H}x{W} a={a} i={inclination_deg}° ===")
    t0 = time.perf_counter()
    n_disk = 0
    for iy in range(H):
        for ix in range(W):
            alpha = float(alpha_grid[iy, ix])
            beta = float(beta_grid[iy, ix])
            hit = trace_single_ray(
                alpha, beta, a=a, inclination_deg=inclination_deg, r_obs=r_obs,
            )
            if hit.status != "disk" or hit.radius is None:
                continue
            n_disk += 1
            r_em = float(hit.radius)
            theta_em = math.pi / 2.0
            f = emitted_polarization_vector_toroidal(
                r_em=r_em, theta_em=theta_em, p_state=hit.state, a=a,
            )
            kappa = walker_penrose_complex(hit.state, f, a)
            chi_obs = decode_observer_evpa(
                kappa, alpha, beta, a, inclination_rad,
            )
            # Emission EVPA in the fluid frame: for a toroidal B field +
            # synchrotron, the E-vector is ⊥ to B; in our chosen basis the
            # fluid-frame EVPA can be taken as 0 (we baked the projection
            # into f), so the observer EVPA is directly chi_obs.
            chi_em = 0.0
            I_obs, Q_obs, U_obs = rotate_stokes_to_observer(
                intensity=hit.intensity, chi_em=chi_em, chi_obs=chi_obs,
                pi_linear=PI_LINEAR,
            )
            I_map[iy, ix] = I_obs
            Q_map[iy, ix] = Q_obs
            U_map[iy, ix] = U_obs
            chi_em_map[iy, ix] = chi_em
            chi_obs_map[iy, ix] = chi_obs
            kappa_re_map[iy, ix] = kappa.real
            kappa_im_map[iy, ix] = kappa.imag
            hit_mask[iy, ix] = True
        if (iy + 1) % 8 == 0:
            print(f"  row {iy+1}/{H}  disk-hit so far: {n_disk}  elapsed {time.perf_counter()-t0:.1f}s")

    elapsed = time.perf_counter() - t0
    pi_obs = np.where(I_map > 0, np.hypot(Q_map, U_map) / np.maximum(I_map, 1e-12), 0.0)
    pi_obs_mean = float(np.mean(pi_obs[hit_mask])) if hit_mask.any() else 0.0
    pi_obs_max = float(np.max(pi_obs)) if hit_mask.any() else 0.0

    print(f"\nTotal disk hits: {n_disk}/{H*W}  Π_obs mean={pi_obs_mean:.3f} max={pi_obs_max:.3f}  ({elapsed:.1f}s)")

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    out = {
        "resolution": RESOLUTION,
        "spin": a,
        "inclination_deg": inclination_deg,
        "pi_linear": PI_LINEAR,
        "disk_hit_pixels": n_disk,
        "pi_obs_mean": pi_obs_mean,
        "pi_obs_max": pi_obs_max,
        "stokes_I_max": float(np.max(I_map)),
        "stokes_Q_minmax": [float(np.min(Q_map)), float(np.max(Q_map))],
        "stokes_U_minmax": [float(np.min(U_map)), float(np.max(U_map))],
        "elapsed_s": elapsed,
    }
    with open("results/polarization_demo.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved results/polarization_demo.json")

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable, skipping figures")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    im0 = axes[0].imshow(I_map, origin="lower", cmap="inferno",
                         extent=(alpha_grid.min(), alpha_grid.max(),
                                 beta_grid.min(), beta_grid.max()))
    axes[0].set_title("Stokes I (intensity)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    qmax = max(abs(Q_map.min()), abs(Q_map.max()), 1e-12)
    im1 = axes[1].imshow(Q_map, origin="lower", cmap="RdBu_r",
                         vmin=-qmax, vmax=qmax,
                         extent=(alpha_grid.min(), alpha_grid.max(),
                                 beta_grid.min(), beta_grid.max()))
    axes[1].set_title("Stokes Q")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    umax = max(abs(U_map.min()), abs(U_map.max()), 1e-12)
    im2 = axes[2].imshow(U_map, origin="lower", cmap="RdBu_r",
                         vmin=-umax, vmax=umax,
                         extent=(alpha_grid.min(), alpha_grid.max(),
                                 beta_grid.min(), beta_grid.max()))
    axes[2].set_title("Stokes U")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Polarization demo (a={a}, i={inclination_deg:.0f}°, Π_em={PI_LINEAR})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figures/polarization_stokes_qu.png", dpi=150)
    plt.close()
    print("Saved figures/polarization_stokes_qu.png")

    # EVPA quiver: each disk-hit pixel gets a small line segment along
    # the local E-vector, scaled by Π_obs.
    fig2, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(I_map, origin="lower", cmap="inferno",
              extent=(alpha_grid.min(), alpha_grid.max(),
                      beta_grid.min(), beta_grid.max()),
              alpha=0.7)
    step = max(1, RESOLUTION // 24)
    for iy in range(0, H, step):
        for ix in range(0, W, step):
            if not hit_mask[iy, ix]:
                continue
            chi = chi_obs_map[iy, ix]
            pi_pix = pi_obs[iy, ix]
            length = pi_pix * 1.5  # scale for visibility
            cx = float(alpha_grid[iy, ix])
            cy = float(beta_grid[iy, ix])
            dx = length * math.cos(chi)
            dy = length * math.sin(chi)
            ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], "c-", lw=0.7, alpha=0.9)
    ax.set_xlabel(r"$\alpha$ [M]")
    ax.set_ylabel(r"$\beta$ [M]")
    ax.set_title(f"EVPA tickmarks over Stokes I  (a={a}, i={inclination_deg:.0f}°)")
    plt.tight_layout()
    plt.savefig("figures/polarization_evpa_quiver.png", dpi=150)
    plt.close()
    print("Saved figures/polarization_evpa_quiver.png")


if __name__ == "__main__":
    main()
