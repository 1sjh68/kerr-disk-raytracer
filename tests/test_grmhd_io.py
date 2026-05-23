"""Tests for src/grmhd_io.py (GRMHD ingestion stub)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.grmhd_io import (
    IHARMFluidSnapshot,
    emission_coefficient_thermal,
    synthetic_thin_disk_fluid,
)


def test_synthetic_snapshot_shape() -> None:
    snap = synthetic_thin_disk_fluid(spin=0.7, n_r=32, n_theta=16, n_phi=16)
    assert snap.shape == (32, 16, 16)
    assert isinstance(snap, IHARMFluidSnapshot)
    assert snap.r.shape == (32,)
    assert snap.theta.shape == (16,)
    assert snap.phi.shape == (16,)
    assert snap.u_con.shape == (4, 32, 16, 16)
    assert snap.B_con.shape == (4, 32, 16, 16)


def test_synthetic_density_decays_with_radius() -> None:
    snap = synthetic_thin_disk_fluid(spin=0.7, n_r=32, n_theta=16, n_phi=8)
    # Equatorial slice
    eq = snap.theta.size // 2
    rho_at_phi0 = snap.rho[:, eq, 0]
    inner_third = rho_at_phi0[:10].mean()
    outer_third = rho_at_phi0[-10:].mean()
    assert inner_third > outer_third, "density should peak inward, not outward"


def test_synthetic_temperature_inner_hot() -> None:
    snap = synthetic_thin_disk_fluid(spin=0.7, n_r=32, n_theta=16, n_phi=8)
    eq = snap.theta.size // 2
    T_inner = snap.T_e[0, eq, 0]
    T_outer = snap.T_e[-1, eq, 0]
    assert T_inner > T_outer, "inner disk should be hotter"


def test_emission_coefficient_nonneg_finite() -> None:
    snap = synthetic_thin_disk_fluid(spin=0.5, n_r=16, n_theta=8, n_phi=4)
    j = emission_coefficient_thermal(snap.rho, snap.T_e, nu_obs_hz=230e9)
    assert np.all(j >= 0)
    assert np.all(np.isfinite(j))


def test_emission_coefficient_zero_at_zero_density() -> None:
    rho = np.zeros((4, 4, 4))
    T_e = np.full_like(rho, 1.0e7)
    j = emission_coefficient_thermal(rho, T_e)
    assert np.allclose(j, 0.0)


def test_load_iharm_missing_file_raises() -> None:
    from src.grmhd_io import load_iharm_snapshot
    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_iharm_snapshot("nonexistent_snapshot.h5")
