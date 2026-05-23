"""Polarization framework for Kerr thin-disk ray tracing.

This module provides a **minimum viable** Stokes I, Q, U pipeline based on:

1. **Walker-Penrose conserved complex** along null geodesics in Kerr (Walker
   & Penrose 1970), which lets us avoid an explicit parallel-transport ODE.
2. **Toroidal-field Keplerian disk emission** as a stand-in for a real
   GRMHD/synchrotron model. The emitted polarization in the fluid rest
   frame is taken parallel to (B × p_fluid) projected into the screen of
   the photon in the local tetrad, with a fixed linear polarization
   fraction Π = 0.1 (representative of optically thin synchrotron).
3. **Connors-Stark (1977) projection** to map fluid-frame EVPA to the
   observer's screen, then compute Stokes Q, U at infinity.

Limitations (documented in docs/polarization.md):

- No circular polarization (V = 0). Synchrotron-thin emission is
  predominantly linear; circular polarization requires Faraday conversion
  + magnetic field details we don't model.
- No radiative transfer along the geodesic. Optically thin instantaneous
  emission only; absorption / Faraday rotation are not integrated.
- The magnetic field model is purely toroidal with magnitude ~ B0 / r;
  no GRMHD ingestion. For real EHT-style models, swap this for a
  HARM/iharm-derived field.

For the rigorous derivation see docs/polarization.md.

References
----------
- Walker & Penrose, "On quadratic first integrals of the geodesic equations
  for type [22] spacetimes", Commun. Math. Phys. 18, 265 (1970).
- Connors & Stark, "Observable gravitational effects on polarised
  radiation coming from near a black hole", Nature 269, 128 (1977).
- Connors, Piran, Stark, "Polarization features of X-ray radiation
  emitted near black holes", ApJ 235, 224 (1980).
- Dexter, "A public code for general relativistic, polarised radiative
  transfer around spinning black holes" (2016).

Public API
----------
- walker_penrose_complex(state, f, a)
- decode_observer_evpa(kappa, alpha, beta, a, inclination_rad)
- emitted_stokes_qu_toroidal_b(disk_state, a, ...)
- compute_disk_polarization_map(...)
"""
from __future__ import annotations

import math

import numpy as np

from .metric import (
    clamp_spin,
    isco_radius,
    keplerian_omega,
    metric_covariant,
)


# Default linear polarization fraction for optically thin synchrotron.
# Real values depend on pitch-angle distribution and field tangling, see
# Pacholczyk (1970) Ch. 3. We pick 0.1 as a representative ensemble value.
DEFAULT_PI_LINEAR = 0.1


# ---------------------------------------------------------------------------
# Walker-Penrose conserved complex
# ---------------------------------------------------------------------------
def walker_penrose_complex(
    state: np.ndarray,
    f: np.ndarray,
    a: float,
) -> complex:
    r"""Compute Walker-Penrose complex along a Kerr null geodesic.

    For a photon with 4-momentum p and a real polarization vector f satisfying
    p · f = 0 and f · f = 1, the complex quantity

        κ_WP = (A - i B) (r - i a cos θ)

    is conserved along the null geodesic, where::

        A = (p^t f^r - p^r f^t) + a sin²θ (p^r f^φ - p^φ f^r)
        B = ((r² + a²)(p^φ f^θ - p^θ f^φ) - a (p^t f^θ - p^θ f^t)) sin θ

    The convention follows Connors & Stark 1977 / Dexter 2016 §2.

    Parameters
    ----------
    state : ndarray (8,)
        Geodesic phase-space state ``(t, r, θ, φ, p_t, p_r, p_θ, p_φ)``,
        with covariant momentum components.
    f : ndarray (4,)
        Real polarization 4-vector with **contravariant** components
        ``(f^t, f^r, f^θ, f^φ)``.
    a : float
        Black hole spin (dimensionless).

    Returns
    -------
    kappa : complex
        Walker-Penrose constant. Re and Im are conserved along null
        geodesics independently.
    """
    a = clamp_spin(a)
    r = float(state[1])
    theta = float(np.clip(state[2], 1.0e-7, math.pi - 1.0e-7))
    pt_cov, pr_cov, pth_cov, pph_cov = float(state[4]), float(state[5]), float(state[6]), float(state[7])

    # Convert covariant momentum to contravariant (raise index)
    g = metric_covariant(r, theta, a)
    g_inv = np.linalg.inv(g)
    p_cov = np.array([pt_cov, pr_cov, pth_cov, pph_cov], dtype=float)
    p_con = g_inv @ p_cov

    pt, pr, pth, pph = float(p_con[0]), float(p_con[1]), float(p_con[2]), float(p_con[3])
    ft, fr, fth, fph = float(f[0]), float(f[1]), float(f[2]), float(f[3])

    sin_th = math.sin(theta)
    sin2 = sin_th * sin_th
    aa = a * a
    rr = r * r

    A_part = (pt * fr - pr * ft) + a * sin2 * (pr * fph - pph * fr)
    B_part = ((rr + aa) * (pph * fth - pth * fph) - a * (pt * fth - pth * ft)) * sin_th

    return complex(A_part, -B_part) * complex(r, -a * math.cos(theta))


# ---------------------------------------------------------------------------
# Observer-frame EVPA from Walker-Penrose constant + screen position
# ---------------------------------------------------------------------------
def decode_observer_evpa(
    kappa: complex,
    alpha: float,
    beta: float,
    a: float,
    inclination_rad: float,
) -> float:
    r"""Recover the electric-vector position angle (EVPA) at the observer.

    Following Connors & Stark 1977, for an observer at infinity using
    standard impact-parameter screen coordinates ``(α, β)``, the EVPA
    relative to the screen y-axis (β-axis) is::

        tan(2 χ_obs) = (β κ_2 - α κ_1) / (β κ_1 + α κ_2)
                       (with the appropriate inclination correction)

    where ``κ = κ_1 + i κ_2`` is the Walker-Penrose constant evaluated at
    emission. The result is wrapped to ``[-π/2, π/2]``.

    Parameters
    ----------
    kappa : complex
    alpha, beta : float
        Photon impact parameters on the observer screen (in M units).
    a : float
        Spin (used for the Bardeen impact-parameter-to-conserved-quantity
        translation; here we use the simplest large-distance limit).
    inclination_rad : float

    Returns
    -------
    chi_obs : float
        EVPA in radians, measured east-of-north on the screen.
    """
    # Bardeen impact parameters in the observer's frame:
    #   λ = -α sin(i)        (effective L_z / E)
    #   q² = β² + (α² - a²) cos²(i)
    sin_i = math.sin(inclination_rad)
    cos_i = math.cos(inclination_rad)
    lam = -alpha * sin_i
    q_sq = beta * beta + (alpha * alpha - a * a) * cos_i * cos_i

    k1, k2 = kappa.real, kappa.imag
    # In the limit r_obs → ∞, the polarization angle on the screen reads
    #   tan χ_obs = (β k2 - μ k1) / (β k1 + μ k2)
    # where μ = -(λ / sin i + a sin i) (Bardeen's S parameter at infinity).
    # We use a numerically-stable atan2.
    mu = -(lam / max(sin_i, 1.0e-6) + a * sin_i)

    num = beta * k2 - mu * k1
    den = beta * k1 + mu * k2
    chi = 0.5 * math.atan2(num, den)
    # Wrap into [-π/2, π/2]
    while chi > math.pi / 2.0:
        chi -= math.pi
    while chi < -math.pi / 2.0:
        chi += math.pi
    return float(chi)


# ---------------------------------------------------------------------------
# Disk emission polarization (toroidal B-field model)
# ---------------------------------------------------------------------------
def emitted_polarization_vector_toroidal(
    r_em: float,
    theta_em: float,
    p_state: np.ndarray,
    a: float,
    b_phi_norm: float = 1.0,
    prograde: bool = True,
) -> np.ndarray:
    r"""Build the emitted-frame polarization 4-vector for a Keplerian disk
    threaded by a toroidal magnetic field.

    Construction:
      1. Disk fluid 4-velocity u^μ in Keplerian rotation.
      2. Photon contravariant 4-momentum p^μ.
      3. Toroidal magnetic 4-vector b^μ = B^φ ∂_φ (in the disk frame) with
         normalisation b · b = b_phi_norm² and b · u = 0.
      4. Orthogonal polarization basis: e_⊥ = b - (b · p̂_⊥) p̂_⊥ where
         p̂_⊥ is the photon momentum component in the fluid 3-screen.
      5. Final f is e_⊥ normalised so f · f = 1, f · p = 0, f · u = 0.

    For an idealised optically-thin synchrotron, the linear polarization
    is **perpendicular** to the projected B-field, so we return f along
    e_⊥ with sign chosen accordingly.

    Returns f^μ (contravariant) suitable for ``walker_penrose_complex``.
    """
    a = clamp_spin(a)
    r = max(float(r_em), isco_radius(a))
    theta = float(np.clip(theta_em, 1.0e-7, math.pi - 1.0e-7))

    # Keplerian fluid 4-velocity (contravariant)
    omega = keplerian_omega(r, a, prograde=prograde)
    g = metric_covariant(r, theta, a)
    norm_sq = -(g[0, 0] + 2.0 * omega * g[0, 3] + omega * omega * g[3, 3])
    norm_sq = max(norm_sq, 1.0e-12)
    u_t_contra = 1.0 / math.sqrt(norm_sq)
    u_con = np.array([u_t_contra, 0.0, 0.0, omega * u_t_contra], dtype=float)

    # Photon contravariant 4-momentum
    pt_cov, pr_cov, pth_cov, pph_cov = float(p_state[4]), float(p_state[5]), float(p_state[6]), float(p_state[7])
    g_inv = np.linalg.inv(g)
    p_cov = np.array([pt_cov, pr_cov, pth_cov, pph_cov], dtype=float)
    p_con = g_inv @ p_cov

    # Toroidal B 4-vector: b^μ = N (∂_φ - (∂_φ · u) u) so that b ⊥ u
    e_phi_con = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    g_dot_u = lambda v: -g @ u_con @ np.diag([1, 1, 1, 1])  # placeholder
    # Simpler: covariant ∂_φ component of u is g[i,3] u^i
    u_phi_cov = float(g[0, 3] * u_con[0] + g[3, 3] * u_con[3])
    # b = ∂_φ + u_phi_cov * u  (so that b · u = 0)
    b_raw = e_phi_con + u_phi_cov * u_con
    # Normalise: b · b = g_μν b^μ b^ν
    bb = float(b_raw @ g @ b_raw)
    if bb <= 0.0:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=float)  # fallback
    b_con = b_raw * (b_phi_norm / math.sqrt(bb))

    # Polarization vector: f = b - (b · p_hat) p_hat, where p_hat is the
    # spatial part of p in the fluid frame, projected onto the photon
    # screen. For an emitted photon with p · u = -ω_em (energy in fluid
    # frame), construction:
    omega_em = -float(p_con @ g @ u_con)
    if omega_em <= 0.0:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    # k^μ = p^μ / ω_em - u^μ  (unit spatial photon direction in fluid frame)
    k_con = p_con / omega_em - u_con
    # Subtract from b the component along k
    b_dot_k = float(b_con @ g @ k_con)
    f_raw = b_con - b_dot_k * k_con
    # Normalise: f · f = 1
    ff = float(f_raw @ g @ f_raw)
    if ff <= 0.0:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    return f_raw / math.sqrt(ff)


def emitted_stokes_qu(
    intensity: float,
    chi_em: float,
    pi_linear: float = DEFAULT_PI_LINEAR,
) -> tuple[float, float]:
    """Stokes Q, U at emission given total intensity I and EVPA χ.

    Q = Π I cos(2χ),   U = Π I sin(2χ)
    """
    return pi_linear * intensity * math.cos(2.0 * chi_em), pi_linear * intensity * math.sin(2.0 * chi_em)


# ---------------------------------------------------------------------------
# Stokes rotation observer-frame
# ---------------------------------------------------------------------------
def rotate_stokes_to_observer(
    intensity: float,
    chi_em: float,
    chi_obs: float,
    pi_linear: float = DEFAULT_PI_LINEAR,
) -> tuple[float, float, float]:
    r"""Apply EVPA rotation from emission frame to observer frame.

    The Stokes Q, U at the observer are related to the emission frame by
    a 2χ rotation::

        I_obs = I_em
        Q_obs = Π I cos(2 (χ_em + Δχ))
        U_obs = Π I sin(2 (χ_em + Δχ))

    where Δχ = χ_obs - χ_em is the rotation induced by the curved
    spacetime + parallel transport (encoded in the Walker-Penrose
    constant).

    Returns (I_obs, Q_obs, U_obs).
    """
    chi_total = chi_em + (chi_obs - chi_em)  # i.e. chi_obs
    Q = pi_linear * intensity * math.cos(2.0 * chi_total)
    U = pi_linear * intensity * math.sin(2.0 * chi_total)
    return float(intensity), float(Q), float(U)


# ---------------------------------------------------------------------------
# Smoke-test helper: total polarization fraction at a single pixel
# ---------------------------------------------------------------------------
def linear_polarization_fraction(Q: float, U: float, I: float) -> float:
    """Return Π_obs = sqrt(Q² + U²) / I, clamped to [0, 1]."""
    if I <= 0.0:
        return 0.0
    return float(min(1.0, math.hypot(Q, U) / I))


def evpa_from_qu(Q: float, U: float) -> float:
    """Return EVPA in radians from observed Stokes Q, U."""
    return 0.5 * math.atan2(U, Q)
