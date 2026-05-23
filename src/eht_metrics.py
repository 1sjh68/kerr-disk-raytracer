"""EHT-style image metrics: ring diameter, asymmetry, photon-ring proxy.

These metrics let us compare the project's thin-disk / synchrotron-stub
images to the EHT M87* / Sgr A* observable summaries (Sec 6 of
research/literature_review.md). They are NOT a substitute for
GRMHD-fitted parameter inference — see docs/polarization.md and
literature_review.md for the scope of validity.

Functions
---------
- ring_diameter        : geometric diameter of high-intensity annulus
- ring_brightness_asymmetry : north / south flux ratio
- photon_ring_position : argmax along radial profile
- summary              : convenience wrapper running all three
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _radial_phi_grids(image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel (r_pix, phi_rad) arrays in image-centred coords."""
    H, W = image_shape
    yy, xx = np.indices((H, W), dtype=float)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    dx = xx - cx
    dy = yy - cy  # numpy y axis points down, but for symmetry analysis sign is consistent
    r_pix = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    return r_pix, phi


def ring_diameter(
    intensity: np.ndarray,
    fov_m: float,
    threshold_frac: float = 0.5,
) -> float:
    """Diameter (in M units) of the bright annulus, defined as the
    median radial position of pixels with intensity >= threshold_frac *
    intensity.max(). Returns ``2 * median_radius_in_M``.

    For a ring image with a clean shadow this gives a value comparable
    to EHT's ring diameter quote.
    """
    if intensity.size == 0 or float(intensity.max()) <= 0.0:
        return 0.0
    r_pix, _ = _radial_phi_grids(intensity.shape)
    H = intensity.shape[0]
    px_per_M = (H - 1) / fov_m  # field-of-view spans fov_m in M units along each axis
    threshold = threshold_frac * float(intensity.max())
    bright = intensity >= threshold
    if not bright.any():
        return 0.0
    radii_M = r_pix[bright] / max(px_per_M, 1.0e-9)
    return float(2.0 * np.median(radii_M))


def ring_brightness_asymmetry(intensity: np.ndarray) -> dict[str, float]:
    """North / south brightness ratio.

    Splits the image into upper (β > 0) and lower (β < 0) halves with
    respect to the optical axis. EHT M87* paper IV reports a
    south-to-north (or equivalent crescent) asymmetry of ≳ 10:1.
    """
    if intensity.size == 0:
        return {"north_flux": 0.0, "south_flux": 0.0, "south_over_north": 0.0}
    H, W = intensity.shape
    cy = (H - 1) / 2.0
    yy = np.arange(H).reshape(H, 1)
    upper = yy < cy   # north (large β)
    lower = yy > cy   # south (small β)
    north = float(intensity[upper.repeat(W, axis=1)].sum())
    south = float(intensity[lower.repeat(W, axis=1)].sum())
    ratio = south / max(north, 1.0e-12)
    return {"north_flux": north, "south_flux": south, "south_over_north": ratio}


def photon_ring_position(
    intensity: np.ndarray,
    fov_m: float,
    n_radial_bins: int = 24,
) -> dict[str, float]:
    """Radial intensity profile and the radius of its maximum (proxy for
    ring brightness peak). Useful as a photon-ring proxy for thin disk."""
    if intensity.size == 0 or float(intensity.max()) <= 0.0:
        return {"peak_radius_M": 0.0, "peak_intensity": 0.0}
    r_pix, _ = _radial_phi_grids(intensity.shape)
    H = intensity.shape[0]
    px_per_M = (H - 1) / fov_m
    r_M = r_pix / max(px_per_M, 1.0e-9)
    r_max = float(r_M.max())
    bin_edges = np.linspace(0.0, r_max, n_radial_bins + 1)
    profile = np.zeros(n_radial_bins, dtype=float)
    counts = np.zeros(n_radial_bins, dtype=float)
    flat_r = r_M.ravel()
    flat_I = intensity.ravel().astype(float)
    idx = np.clip(np.digitize(flat_r, bin_edges) - 1, 0, n_radial_bins - 1)
    np.add.at(profile, idx, flat_I)
    np.add.at(counts, idx, 1.0)
    profile_avg = profile / np.maximum(counts, 1.0)
    peak_bin = int(np.argmax(profile_avg))
    peak_r = 0.5 * (bin_edges[peak_bin] + bin_edges[peak_bin + 1])
    return {
        "peak_radius_M": float(peak_r),
        "peak_intensity": float(profile_avg[peak_bin]),
        "radial_profile": profile_avg.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


def summary(intensity: np.ndarray, fov_m: float) -> dict[str, Any]:
    """Wrapper: ring diameter + asymmetry + photon-ring proxy."""
    out: dict[str, Any] = {
        "image_shape": list(intensity.shape),
        "fov_m": fov_m,
        "intensity_max": float(intensity.max()) if intensity.size else 0.0,
        "intensity_total": float(intensity.sum()) if intensity.size else 0.0,
    }
    out["ring_diameter_M"] = ring_diameter(intensity, fov_m, threshold_frac=0.5)
    out["asymmetry"] = ring_brightness_asymmetry(intensity)
    pr = photon_ring_position(intensity, fov_m)
    out["photon_ring_peak_radius_M"] = pr["peak_radius_M"]
    out["photon_ring_peak_intensity"] = pr["peak_intensity"]
    return out
