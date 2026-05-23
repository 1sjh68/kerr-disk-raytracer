from __future__ import annotations

import numpy as np

from src.camera import initial_photon_state
from src.geodesic import hamiltonian_rhs_for_spin, trace_single_ray
from src.integrators import rk4_step
from src.metric import null_hamiltonian


def test_initial_photon_satisfies_null_constraint() -> None:
    state = initial_photon_state(0.0, 0.0, a=0.3, r_obs=60.0, inclination_deg=45.0)
    assert abs(null_hamiltonian(state, 0.3)) < 1.0e-10


def test_single_small_step_preserves_null_constraint_reasonably() -> None:
    a = 0.3
    state = initial_photon_state(1.0, 0.5, a=a, r_obs=60.0, inclination_deg=45.0)
    next_state = rk4_step(hamiltonian_rhs_for_spin(a), state, 0.05)
    assert np.all(np.isfinite(next_state))
    assert abs(null_hamiltonian(next_state, a)) < 1.0e-6


def test_trace_single_ray_returns_terminal_status() -> None:
    hit = trace_single_ray(
        3.0,
        1.0,
        a=0.2,
        inclination_deg=60.0,
        r_obs=30.0,
        step_size=0.2,
        max_steps=180,
        escape_radius=45.0,
    )
    assert hit.status in {"disk", "captured", "escaped", "max_steps"}
    assert np.all(np.isfinite(hit.state))

