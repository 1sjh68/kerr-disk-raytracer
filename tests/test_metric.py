from __future__ import annotations

import math

import numpy as np

from src.metric import (
    horizon_radius,
    isco_radius,
    metric_covariant,
    metric_contravariant,
    metric_derivatives,
)


def test_schwarzschild_horizon_and_isco() -> None:
    assert abs(horizon_radius(0.0) - 2.0) < 1.0e-12
    assert abs(isco_radius(0.0) - 6.0) < 1.0e-12


def test_high_spin_isco_moves_inward() -> None:
    assert isco_radius(0.9) < isco_radius(0.5) < isco_radius(0.0)


def test_metric_inverse_identity() -> None:
    g = metric_covariant(10.0, math.radians(63.0), 0.7)
    gi = metric_contravariant(10.0, math.radians(63.0), 0.7)
    ident = g @ gi
    assert np.max(np.abs(ident - np.eye(4))) < 1.0e-10


def test_inverse_metric_derivatives_match_finite_difference() -> None:
    r = 8.0
    theta = math.radians(61.0)
    a = 0.7
    dg_dr, dg_dt = metric_derivatives(r, theta, a)
    h = 1.0e-5
    fd_r = (metric_contravariant(r + h, theta, a) - metric_contravariant(r - h, theta, a)) / (2.0 * h)
    fd_t = (metric_contravariant(r, theta + h, a) - metric_contravariant(r, theta - h, a)) / (2.0 * h)
    assert np.max(np.abs(dg_dr - fd_r)) < 1.0e-8
    assert np.max(np.abs(dg_dt - fd_t)) < 1.0e-8
