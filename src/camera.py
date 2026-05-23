from __future__ import annotations

import math

import numpy as np

from .metric import metric_contravariant, metric_covariant


def screen_grid(resolution: int, fov_m: float) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-0.5 * fov_m, 0.5 * fov_m, resolution, dtype=float)
    alpha, beta = np.meshgrid(axis, axis)
    return alpha, beta


def initial_photon_state(
    alpha: float,
    beta: float,
    a: float,
    r_obs: float = 60.0,
    inclination_deg: float = 60.0,
) -> np.ndarray:
    """Approximate far-observer initial photon in BL coordinates.

    The spatial direction is initialized in the local asymptotic frame and then
    converted to covariant momenta. ``p_t`` is solved from the null constraint.
    This is suitable for regression tests and low-resolution MVP tracing; the
    final high-precision version should replace it with a full observer tetrad.
    """

    theta = math.radians(inclination_deg)
    theta = float(np.clip(theta, 1.0e-5, math.pi - 1.0e-5))
    n_phi = alpha / r_obs
    n_theta = beta / r_obs
    transverse2 = n_phi * n_phi + n_theta * n_theta
    if transverse2 >= 0.95:
        scale = math.sqrt(0.95 / transverse2)
        n_phi *= scale
        n_theta *= scale
        transverse2 = n_phi * n_phi + n_theta * n_theta
    n_r = -math.sqrt(max(1.0e-12, 1.0 - transverse2))

    g = metric_covariant(r_obs, theta, a)
    p_r = math.sqrt(g[1, 1]) * n_r
    p_theta = math.sqrt(g[2, 2]) * n_theta
    p_phi = math.sqrt(max(g[3, 3], 1.0e-12)) * n_phi
    gi = metric_contravariant(r_obs, theta, a)
    A = gi[0, 0]
    B = 2.0 * gi[0, 3] * p_phi
    C = gi[1, 1] * p_r * p_r + gi[2, 2] * p_theta * p_theta + gi[3, 3] * p_phi * p_phi
    disc = max(0.0, B * B - 4.0 * A * C)
    roots = [(-B + math.sqrt(disc)) / (2.0 * A), (-B - math.sqrt(disc)) / (2.0 * A)]

    def future_dt(pt: float) -> float:
        return gi[0, 0] * pt + gi[0, 3] * p_phi

    future_roots = [pt for pt in roots if future_dt(pt) > 0.0]
    p_t = min(future_roots) if future_roots else min(roots)
    return np.array([0.0, r_obs, theta, 0.0, p_t, p_r, p_theta, p_phi], dtype=float)

