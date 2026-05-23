from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .camera import initial_photon_state
from .disk import disk_inner_radius, hit_intensity
from .integrators import rk4_step, rk45_step
from .metric import horizon_radius, inverse_metric_terms_and_derivatives, null_hamiltonian


@dataclass(frozen=True)
class RayHit:
    status: str
    steps: int
    state: np.ndarray
    radius: float | None = None
    intensity: float = 0.0
    redshift: float = 0.0
    temperature: float = 0.0
    null_error: float = 0.0
    trajectory: np.ndarray | None = None  # shape (N, 4): (lambda, r, theta, phi)


def hamiltonian_rhs_for_spin(a: float):
    def rhs(state: np.ndarray) -> np.ndarray:
        r = float(state[1])
        theta = float(np.clip(state[2], 1.0e-6, math.pi - 1.0e-6))
        pt, pr, ptheta, pphi = state[4:8]
        (gtt, gtphi, grr, gtheta, gphi), dr_terms, dt_terms = inverse_metric_terms_and_derivatives(
            r, theta, a
        )
        dx_t = gtt * pt + gtphi * pphi
        dx_r = grr * pr
        dx_theta = gtheta * ptheta
        dx_phi = gtphi * pt + gphi * pphi

        def contraction(terms: tuple[float, float, float, float, float]) -> float:
            dgtt, dgtphi, dgrr, dgtheta, dgphi = terms
            with np.errstate(over="ignore", invalid="ignore"):
                value = (
                    dgtt * pt * pt
                    + 2.0 * dgtphi * pt * pphi
                    + dgrr * pr * pr
                    + dgtheta * ptheta * ptheta
                    + dgphi * pphi * pphi
                )
            return float(value) if math.isfinite(float(value)) else math.inf

        dp_r = -0.5 * contraction(dr_terms)
        dp_theta = -0.5 * contraction(dt_terms)
        return np.array([dx_t, dx_r, dx_theta, dx_phi, 0.0, dp_r, dp_theta, 0.0], dtype=float)

    return rhs


def trace_single_ray(
    alpha: float,
    beta: float,
    a: float,
    inclination_deg: float = 60.0,
    r_obs: float = 60.0,
    disk_inner: str | float = "isco",
    disk_outer: float = 28.0,
    emissivity_index: float = 3.0,
    emission_model: str = "power_law",
    step_size: float = 0.35,
    max_steps: int = 700,
    horizon_epsilon: float = 0.05,
    escape_radius: float = 90.0,
    record_trajectory: bool = False,
) -> RayHit:
    state = initial_photon_state(alpha, beta, a, r_obs=r_obs, inclination_deg=inclination_deg)
    rhs = hamiltonian_rhs_for_spin(a)
    r_h = horizon_radius(a)
    r_in = disk_inner_radius(a, disk_inner)
    previous = state.copy()
    traj: list[tuple[float, float, float, float]] | None = (
        [(0.0, float(state[1]), float(state[2]), float(state[3]))] if record_trajectory else None
    )
    affine = 0.0

    for step in range(1, max_steps + 1):
        state = rk4_step(rhs, state, step_size)
        affine += step_size
        r = float(state[1])
        theta = float(state[2])
        if traj is not None and np.all(np.isfinite(state)):
            # sanity-clamp: only record physically reasonable points
            if (
                0.0 <= float(state[1]) <= 1.0e6
                and abs(float(state[2])) <= 100.0
                and abs(float(state[3])) <= 1.0e4
            ):
                traj.append((affine, r, theta, float(state[3])))
        if not np.all(np.isfinite(state)):
            traj_arr = np.asarray(traj, dtype=float) if traj is not None else None
            if np.isfinite(previous[1]) and previous[1] > escape_radius:
                return RayHit("escaped", step, previous, null_error=abs(null_hamiltonian(previous, a)), trajectory=traj_arr)
            near_horizon = np.isfinite(previous[1]) and previous[1] <= r_h + max(2.0, 8.0 * abs(step_size))
            inward = np.isfinite(previous[5]) and previous[5] < 0.0
            status = "captured" if near_horizon and inward else "invalid"
            return RayHit(status, step, previous, null_error=abs(null_hamiltonian(previous, a)), trajectory=traj_arr)
        if r <= r_h + horizon_epsilon:
            traj_arr = np.asarray(traj, dtype=float) if traj is not None else None
            return RayHit("captured", step, state, null_error=abs(null_hamiltonian(state, a)), trajectory=traj_arr)
        if r > escape_radius and step > 3 and state[5] > 0.0:
            traj_arr = np.asarray(traj, dtype=float) if traj is not None else None
            return RayHit("escaped", step, state, null_error=abs(null_hamiltonian(state, a)), trajectory=traj_arr)

        crossed = (previous[2] - math.pi / 2.0) * (theta - math.pi / 2.0) <= 0.0
        if crossed and step > 1:
            denom = theta - previous[2]
            frac = 0.0 if abs(denom) < 1.0e-12 else (math.pi / 2.0 - previous[2]) / denom
            frac = float(np.clip(frac, 0.0, 1.0))
            hit = previous + frac * (state - previous)
            hit_r = float(hit[1])
            if r_in <= hit_r <= disk_outer:
                lam = float(hit[7] / max(1.0e-12, -hit[4]))
                intensity, redshift, temperature = hit_intensity(
                    hit_r,
                    math.pi / 2.0,
                    a,
                    lam,
                    q=emissivity_index,
                    model=emission_model,
                    r_in=r_in,
                )
                traj_arr = np.asarray(traj, dtype=float) if traj is not None else None
                return RayHit(
                    "disk",
                    step,
                    hit,
                    radius=hit_r,
                    intensity=intensity,
                    redshift=redshift,
                    temperature=temperature,
                    null_error=abs(null_hamiltonian(hit, a)),
                    trajectory=traj_arr,
                )
        previous = state.copy()
    traj_arr = np.asarray(traj, dtype=float) if traj is not None else None
    return RayHit("max_steps", max_steps, state, null_error=abs(null_hamiltonian(state, a)), trajectory=traj_arr)




def trace_single_ray_rk45(
    alpha: float,
    beta: float,
    a: float,
    inclination_deg: float = 60.0,
    r_obs: float = 60.0,
    disk_inner: str | float = "isco",
    disk_outer: float = 28.0,
    emissivity_index: float = 3.0,
    emission_model: str = "power_law",
    initial_step: float = 0.35,
    max_steps: int = 5000,
    horizon_epsilon: float = 0.05,
    escape_radius: float = 90.0,
    atol: float = 1.0e-8,
    rtol: float = 1.0e-6,
    record_trajectory: bool = False,
) -> RayHit:
    """Same termination logic as trace_single_ray, but with Dormand-Prince RK45.

    Adaptive step size shrinks automatically near the photon sphere where
    fixed-step RK4 is most error-prone, at the cost of more RHS evaluations
    per step.
    """
    state = initial_photon_state(alpha, beta, a, r_obs=r_obs, inclination_deg=inclination_deg)
    rhs = hamiltonian_rhs_for_spin(a)
    r_h = horizon_radius(a)
    r_in = disk_inner_radius(a, disk_inner)
    previous = state.copy()
    traj: list[tuple[float, float, float, float]] | None = (
        [(0.0, float(state[1]), float(state[2]), float(state[3]))] if record_trajectory else None
    )
    affine = 0.0
    h = float(initial_step)
    accepted_steps = 0
    rejected_steps = 0

    for _step in range(1, max_steps + 1):
        result = rk45_step(rhs, state, h, atol=atol, rtol=rtol)
        if not result.accepted:
            rejected_steps += 1
            h = max(result.h_next, 1.0e-6)
            if h < 1.0e-6:
                break
            continue
        accepted_steps += 1
        new_state = result.y
        affine += h
        h = max(min(result.h_next, 5.0), 1.0e-4)  # clip step size for stability

        r = float(new_state[1])
        theta = float(new_state[2])
        finite_ok = np.all(np.isfinite(new_state))
        if traj is not None and finite_ok and 0.0 <= r <= 1.0e6 and abs(theta) <= 100.0:
            traj.append((affine, r, theta, float(new_state[3])))

        if not finite_ok:
            traj_arr = np.asarray(traj, dtype=float) if traj is not None else None
            if np.isfinite(previous[1]) and previous[1] > escape_radius:
                return RayHit("escaped", accepted_steps, previous,
                              null_error=abs(null_hamiltonian(previous, a)),
                              trajectory=traj_arr)
            near_horizon = np.isfinite(previous[1]) and previous[1] <= r_h + max(2.0, 8.0 * abs(initial_step))
            inward = np.isfinite(previous[5]) and previous[5] < 0.0
            status = "captured" if near_horizon and inward else "invalid"
            return RayHit(status, accepted_steps, previous,
                          null_error=abs(null_hamiltonian(previous, a)),
                          trajectory=traj_arr)

        if r <= r_h + horizon_epsilon:
            traj_arr = np.asarray(traj, dtype=float) if traj is not None else None
            return RayHit("captured", accepted_steps, new_state,
                          null_error=abs(null_hamiltonian(new_state, a)),
                          trajectory=traj_arr)
        if r > escape_radius and accepted_steps > 3 and new_state[5] > 0.0:
            traj_arr = np.asarray(traj, dtype=float) if traj is not None else None
            return RayHit("escaped", accepted_steps, new_state,
                          null_error=abs(null_hamiltonian(new_state, a)),
                          trajectory=traj_arr)

        crossed = (previous[2] - math.pi / 2.0) * (theta - math.pi / 2.0) <= 0.0
        if crossed and accepted_steps > 1:
            denom = theta - previous[2]
            frac = 0.0 if abs(denom) < 1.0e-12 else (math.pi / 2.0 - previous[2]) / denom
            frac = float(np.clip(frac, 0.0, 1.0))
            hit = previous + frac * (new_state - previous)
            hit_r = float(hit[1])
            if r_in <= hit_r <= disk_outer:
                lam = float(hit[7] / max(1.0e-12, -hit[4]))
                intensity, redshift, temperature = hit_intensity(
                    hit_r, math.pi / 2.0, a, lam,
                    q=emissivity_index, model=emission_model, r_in=r_in,
                )
                traj_arr = np.asarray(traj, dtype=float) if traj is not None else None
                return RayHit(
                    "disk", accepted_steps, hit, radius=hit_r,
                    intensity=intensity, redshift=redshift, temperature=temperature,
                    null_error=abs(null_hamiltonian(hit, a)),
                    trajectory=traj_arr,
                )
        previous = new_state.copy()
        state = new_state

    traj_arr = np.asarray(traj, dtype=float) if traj is not None else None
    return RayHit("max_steps", accepted_steps, state,
                  null_error=abs(null_hamiltonian(state, a)),
                  trajectory=traj_arr)
