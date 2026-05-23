from __future__ import annotations

import math

import numpy as np


EPS = 1.0e-12


def clamp_spin(a: float) -> float:
    return float(np.clip(a, -0.999999, 0.999999))


def sigma(r: float | np.ndarray, theta: float | np.ndarray, a: float) -> float | np.ndarray:
    a = clamp_spin(a)
    return np.asarray(r) ** 2 + a * a * np.cos(theta) ** 2


def delta(r: float | np.ndarray, a: float) -> float | np.ndarray:
    a = clamp_spin(a)
    return np.asarray(r) ** 2 - 2.0 * np.asarray(r) + a * a


def horizon_radius(a: float) -> float:
    a = clamp_spin(a)
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a))


def isco_radius(a: float, prograde: bool = True) -> float:
    """Bardeen/Press/Teukolsky ISCO radius in units G=c=M=1."""

    a = clamp_spin(a)
    if not prograde:
        a = -a
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    sign = 1.0 if a >= 0.0 else -1.0
    return 3.0 + z2 - sign * math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def metric_covariant(r: float, theta: float, a: float) -> np.ndarray:
    """Kerr metric g_mu_nu in Boyer-Lindquist coordinates.

    Coordinate order is ``(t, r, theta, phi)`` and the signature is ``(-,+,+,+)``.
    """

    a = clamp_spin(a)
    th = float(np.clip(theta, 1.0e-7, math.pi - 1.0e-7))
    rr = max(float(r), horizon_radius(a) + 1.0e-7)
    sig = float(sigma(rr, th, a))
    dele = float(delta(rr, a))
    sin2 = math.sin(th) ** 2

    g = np.zeros((4, 4), dtype=float)
    g[0, 0] = -(1.0 - 2.0 * rr / sig)
    g[0, 3] = g[3, 0] = -2.0 * a * rr * sin2 / sig
    g[1, 1] = sig / max(dele, EPS)
    g[2, 2] = sig
    g[3, 3] = (rr * rr + a * a + 2.0 * a * a * rr * sin2 / sig) * sin2
    return g


def metric_contravariant(r: float, theta: float, a: float) -> np.ndarray:
    """Closed-form inverse Kerr metric in Boyer-Lindquist coordinates."""

    a = clamp_spin(a)
    th = float(np.clip(theta, 1.0e-7, math.pi - 1.0e-7))
    rr = max(float(r), horizon_radius(a) + 1.0e-7)
    sig = float(sigma(rr, th, a))
    dele = max(float(delta(rr, a)), EPS)
    sin2 = max(math.sin(th) ** 2, EPS)
    r2_a2 = rr * rr + a * a
    aa = a * a
    big_a = r2_a2 * r2_a2 - aa * dele * sin2
    denom = sig * dele

    gi = np.zeros((4, 4), dtype=float)
    gi[0, 0] = -big_a / denom
    gi[0, 3] = gi[3, 0] = -2.0 * a * rr / denom
    gi[1, 1] = dele / sig
    gi[2, 2] = 1.0 / sig
    gi[3, 3] = (dele - aa * sin2) / (denom * sin2)
    return gi


def metric_derivatives(
    r: float, theta: float, a: float, step: float = 1.0e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic derivatives of inverse metric with respect to r/theta."""

    del step
    a = clamp_spin(a)
    th = float(np.clip(theta, 1.0e-7, math.pi - 1.0e-7))
    rr = max(float(r), horizon_radius(a) + 1.0e-7)
    aa = a * a
    s = math.sin(th)
    c = math.cos(th)
    sin2 = max(s * s, EPS)
    dsin2_dt = 2.0 * s * c

    sig = rr * rr + aa * c * c
    sig_r = 2.0 * rr
    sig_t = -2.0 * aa * s * c

    dele = max(rr * rr - 2.0 * rr + aa, EPS)
    dele_r = 2.0 * rr - 2.0

    r2_a2 = rr * rr + aa
    big_a = r2_a2 * r2_a2 - aa * dele * sin2
    big_a_r = 4.0 * rr * r2_a2 - aa * dele_r * sin2
    big_a_t = -aa * dele * dsin2_dt

    denom = sig * dele
    denom_r = sig_r * dele + sig * dele_r
    denom_t = sig_t * dele
    denom2 = denom * denom

    dg_dr = np.zeros((4, 4), dtype=float)
    dg_dt = np.zeros((4, 4), dtype=float)

    dg_dr[0, 0] = -((big_a_r * denom - big_a * denom_r) / denom2)
    dg_dt[0, 0] = -((big_a_t * denom - big_a * denom_t) / denom2)

    gtphi_n = -2.0 * a * rr
    gtphi_n_r = -2.0 * a
    dg_dr[0, 3] = dg_dr[3, 0] = (gtphi_n_r * denom - gtphi_n * denom_r) / denom2
    dg_dt[0, 3] = dg_dt[3, 0] = -(gtphi_n * denom_t) / denom2

    sig2 = sig * sig
    dg_dr[1, 1] = (dele_r * sig - dele * sig_r) / sig2
    dg_dt[1, 1] = -(dele * sig_t) / sig2

    dg_dr[2, 2] = -sig_r / sig2
    dg_dt[2, 2] = -sig_t / sig2

    gph_n = dele - aa * sin2
    gph_n_r = dele_r
    gph_n_t = -aa * dsin2_dt
    gph_d = denom * sin2
    gph_d_r = denom_r * sin2
    gph_d_t = denom_t * sin2 + denom * dsin2_dt
    gph_d2 = gph_d * gph_d
    dg_dr[3, 3] = (gph_n_r * gph_d - gph_n * gph_d_r) / gph_d2
    dg_dt[3, 3] = (gph_n_t * gph_d - gph_n * gph_d_t) / gph_d2
    return dg_dr, dg_dt


def inverse_metric_terms_and_derivatives(
    r: float,
    theta: float,
    a: float,
) -> tuple[tuple[float, float, float, float, float], tuple[float, float, float, float, float], tuple[float, float, float, float, float]]:
    """Return non-zero inverse metric terms and r/theta derivatives.

    Terms are ordered as ``(gtt, gtphi, grr, gthetatheta, gphiphi)``.
    This scalar form avoids allocating 4x4 matrices inside tight ray-tracing
    loops.
    """

    a = clamp_spin(a)
    th = float(np.clip(theta, 1.0e-7, math.pi - 1.0e-7))
    rr = max(float(r), horizon_radius(a) + 1.0e-7)
    aa = a * a
    s = math.sin(th)
    c = math.cos(th)
    sin2 = max(s * s, EPS)
    dsin2_dt = 2.0 * s * c

    sig = rr * rr + aa * c * c
    sig_r = 2.0 * rr
    sig_t = -2.0 * aa * s * c
    dele = max(rr * rr - 2.0 * rr + aa, EPS)
    dele_r = 2.0 * rr - 2.0
    r2_a2 = rr * rr + aa
    big_a = r2_a2 * r2_a2 - aa * dele * sin2
    big_a_r = 4.0 * rr * r2_a2 - aa * dele_r * sin2
    big_a_t = -aa * dele * dsin2_dt

    denom = sig * dele
    denom_r = sig_r * dele + sig * dele_r
    denom_t = sig_t * dele
    denom2 = denom * denom
    sig2 = sig * sig

    gtt = -big_a / denom
    gtphi_n = -2.0 * a * rr
    gtphi = gtphi_n / denom
    grr = dele / sig
    gtheta = 1.0 / sig
    gph_n = dele - aa * sin2
    gph_d = denom * sin2
    gphi = gph_n / gph_d

    dgtt_r = -((big_a_r * denom - big_a * denom_r) / denom2)
    dgtt_t = -((big_a_t * denom - big_a * denom_t) / denom2)
    gtphi_n_r = -2.0 * a
    dgtphi_r = (gtphi_n_r * denom - gtphi_n * denom_r) / denom2
    dgtphi_t = -(gtphi_n * denom_t) / denom2
    dgrr_r = (dele_r * sig - dele * sig_r) / sig2
    dgrr_t = -(dele * sig_t) / sig2
    dgtheta_r = -sig_r / sig2
    dgtheta_t = -sig_t / sig2
    gph_n_r = dele_r
    gph_n_t = -aa * dsin2_dt
    gph_d_r = denom_r * sin2
    gph_d_t = denom_t * sin2 + denom * dsin2_dt
    gph_d2 = gph_d * gph_d
    dgphi_r = (gph_n_r * gph_d - gph_n * gph_d_r) / gph_d2
    dgphi_t = (gph_n_t * gph_d - gph_n * gph_d_t) / gph_d2

    values = (gtt, gtphi, grr, gtheta, gphi)
    dr_terms = (dgtt_r, dgtphi_r, dgrr_r, dgtheta_r, dgphi_r)
    dt_terms = (dgtt_t, dgtphi_t, dgrr_t, dgtheta_t, dgphi_t)
    return values, dr_terms, dt_terms


def null_hamiltonian(state: np.ndarray, a: float) -> float:
    r = float(state[1])
    theta = float(state[2])
    pt, pr, ptheta, pphi = state[4:8]
    with np.errstate(over="ignore", invalid="ignore"):
        gtt, gtphi, grr, gtheta, gphi = inverse_metric_terms_and_derivatives(r, theta, a)[0]
        value = 0.5 * float(
            gtt * pt * pt
            + 2.0 * gtphi * pt * pphi
            + grr * pr * pr
            + gtheta * ptheta * ptheta
            + gphi * pphi * pphi
        )
    return value if math.isfinite(value) else math.inf


def keplerian_omega(r: float, a: float, prograde: bool = True) -> float:
    aa = clamp_spin(a)
    sign = 1.0 if prograde else -1.0
    return sign / (r ** 1.5 + sign * aa)


def keplerian_u_t(r: float, theta: float, a: float, prograde: bool = True) -> float:
    g = metric_covariant(r, theta, a)
    omega = keplerian_omega(r, a, prograde=prograde)
    norm = -(g[0, 0] + 2.0 * omega * g[0, 3] + omega * omega * g[3, 3])
    return 1.0 / math.sqrt(max(norm, EPS))


def redshift_factor_from_lambda(
    r: float, theta: float, a: float, lambda_photon: float, prograde: bool = True
) -> float:
    omega = keplerian_omega(r, a, prograde=prograde)
    u_t = keplerian_u_t(r, theta, a, prograde=prograde)
    denom = u_t * max(0.05, 1.0 - omega * lambda_photon)
    return float(np.clip(1.0 / denom, 0.02, 8.0))
