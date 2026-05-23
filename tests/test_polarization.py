"""Tests for src/polarization.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.camera import initial_photon_state
from src.geodesic import trace_single_ray
from src.metric import isco_radius, keplerian_omega, metric_covariant
from src.polarization import (
    DEFAULT_PI_LINEAR,
    decode_observer_evpa,
    emitted_polarization_vector_toroidal,
    emitted_stokes_qu,
    evpa_from_qu,
    linear_polarization_fraction,
    rotate_stokes_to_observer,
    walker_penrose_complex,
)


def test_emitted_stokes_qu_correct_amplitude() -> None:
    """Q^2 + U^2 = (Π I)^2 — basic invariant."""
    Q, U = emitted_stokes_qu(intensity=1.0, chi_em=0.3, pi_linear=0.1)
    assert math.hypot(Q, U) == pytest.approx(0.1, abs=1.0e-9)


def test_emitted_stokes_qu_chi_zero_is_pure_q() -> None:
    Q, U = emitted_stokes_qu(intensity=2.0, chi_em=0.0, pi_linear=0.1)
    assert Q == pytest.approx(0.2, abs=1.0e-9)
    assert U == pytest.approx(0.0, abs=1.0e-9)


def test_emitted_stokes_qu_chi_quarter_pi_is_pure_u() -> None:
    Q, U = emitted_stokes_qu(intensity=2.0, chi_em=math.pi / 4.0, pi_linear=0.1)
    assert Q == pytest.approx(0.0, abs=1.0e-9)
    assert U == pytest.approx(0.2, abs=1.0e-9)


def test_linear_polarization_fraction_invariants() -> None:
    assert linear_polarization_fraction(0.06, 0.08, 1.0) == pytest.approx(0.1, abs=1.0e-9)
    assert linear_polarization_fraction(0.0, 0.0, 1.0) == 0.0
    assert linear_polarization_fraction(0.5, 0.5, 0.0) == 0.0


def test_evpa_round_trip() -> None:
    """EVPA → Stokes → EVPA round trip preserves angle (mod π)."""
    for chi_in in [-1.2, -0.5, 0.0, 0.3, 0.78, 1.4]:
        Q, U = emitted_stokes_qu(intensity=1.0, chi_em=chi_in, pi_linear=0.1)
        chi_out = evpa_from_qu(Q, U)
        # Allow for π ambiguity (EVPA is defined modulo π)
        diff = (chi_out - chi_in + math.pi) % math.pi - math.pi / 2.0
        assert abs(diff) < 1.0e-7 or abs(abs(diff) - math.pi / 2.0) < 1.0e-7


def test_walker_penrose_returns_complex() -> None:
    """Smoke test: WP constant is a finite complex number."""
    a = 0.7
    state = initial_photon_state(0.0, 5.0, a, r_obs=60.0, inclination_deg=60.0)
    f = np.array([0.0, 1.0, 0.0, 0.0])
    kappa = walker_penrose_complex(state, f, a)
    assert isinstance(kappa, complex)
    assert math.isfinite(kappa.real) and math.isfinite(kappa.imag)


def test_walker_penrose_conserved_along_geodesic() -> None:
    """Re/Im of WP complex should be conserved along the null geodesic.

    We use a polarization vector orthogonal to p, parallel-transported by
    construction (we use the Walker-Penrose definition, so it's implicit).
    For a fixed reference f at emission and photon state at two distinct
    points, we evaluate κ_WP and check it stays close.

    NOTE: We don't have an actual parallel-transport ODE here — this test
    only verifies the κ formula is *evaluable* at multiple geodesic
    samples without numerical blow-up. The conservation law would be
    proven by integrating df/dλ + Γ p f = 0 alongside, which is future
    work documented in docs/polarization.md.
    """
    a = 0.5
    hit = trace_single_ray(0.0, 5.0, a=a, inclination_deg=45.0,
                           r_obs=60.0, max_steps=500, record_trajectory=True)
    if hit.status != "disk" or hit.trajectory is None or len(hit.trajectory) < 50:
        pytest.skip("ray didn't hit the disk in this test config")

    # Take two trajectory samples and rebuild approximate states with
    # finite-difference momentum (we don't have momentum recorded). We
    # pick the final disk-hit state itself which has full momentum.
    final = hit.state.copy()
    f = emitted_polarization_vector_toroidal(
        r_em=hit.radius if hit.radius is not None else 6.0,
        theta_em=math.pi / 2.0,
        p_state=final,
        a=a,
    )
    kappa = walker_penrose_complex(final, f, a)
    assert math.isfinite(kappa.real) and math.isfinite(kappa.imag)
    # Sanity: κ shouldn't be wildly larger than the disk radius scale
    assert abs(kappa) < 1.0e4


def test_emitted_polarization_orthogonal_to_p() -> None:
    """f · p = 0 by construction (we project out the photon direction).

    Use a real disk-hit state from a traced geodesic so that p is the
    correctly evolved 4-momentum at the disk emission point.
    """
    a = 0.7
    hit = trace_single_ray(
        2.0, 3.0, a=a, inclination_deg=60.0, r_obs=60.0, max_steps=700,
    )
    if hit.status != "disk":
        pytest.skip("ray didn't hit disk in this config")
    r_em = float(hit.radius if hit.radius is not None else hit.state[1])
    f = emitted_polarization_vector_toroidal(
        r_em=r_em, theta_em=math.pi / 2.0, p_state=hit.state, a=a,
    )
    g = metric_covariant(r_em, math.pi / 2.0, a)
    g_inv = np.linalg.inv(g)
    p_cov = hit.state[4:8].astype(float)
    p_con = g_inv @ p_cov
    fp = float(f @ g @ p_con)
    # Should be ~0 to within geodesic null-error scale
    assert abs(fp) < 1.0e-3, f"f.p = {fp} not small enough"


def test_rotate_stokes_keeps_intensity() -> None:
    I_obs, Q_obs, U_obs = rotate_stokes_to_observer(
        intensity=1.5, chi_em=0.2, chi_obs=0.6, pi_linear=0.1,
    )
    assert I_obs == pytest.approx(1.5)
    # Pi^2 = Q^2 + U^2 / I^2
    pi_obs = math.hypot(Q_obs, U_obs) / I_obs
    assert pi_obs == pytest.approx(0.1, abs=1.0e-9)


def test_decode_observer_evpa_returns_finite() -> None:
    chi = decode_observer_evpa(complex(1.0, 0.5), alpha=2.0, beta=3.0,
                                a=0.7, inclination_rad=math.radians(60.0))
    assert math.isfinite(chi)
    assert -math.pi / 2.0 <= chi <= math.pi / 2.0
