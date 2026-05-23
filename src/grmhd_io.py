"""GRMHD ingestion stub for HARM / iharm-format snapshots.

This is a **stub** for Phase-11 future expansion. It defines the
expected data interface and provides a synthetic fallback (analytic
Novikov-Thorne disc proxy) so downstream code can be written and
tested without an actual GRMHD snapshot file.

Real HARM/iharm snapshots are written as HDF5 with a layout described
in:
- Gammie, McKinney, Tóth 2003 (HARM)
- Prather et al. 2021 (iharm3d / iharm2d_v3)
- Wong et al. 2022 (PATOKA-format pipeline used by EHT)

When you have a snapshot, install ``h5py`` and call
:func:`load_iharm_snapshot`. The synthetic fallback is selected
automatically when the file is missing or h5py is unavailable.

Public API
----------
- IHARMFluidSnapshot      dataclass with ρ, T, B, u in BL coordinates
- load_iharm_snapshot     read from HDF5 (requires h5py)
- synthetic_thin_disk_fluid  analytic Novikov-Thorne stand-in
- emission_coefficient_thermal  simple frequency-dependent thermal synchrotron
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metric import isco_radius


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class IHARMFluidSnapshot:
    """Snapshot of a GRMHD fluid state in Boyer-Lindquist coordinates.

    All arrays are 3D Cartesian-style ``(N_r, N_theta, N_phi)`` grids
    sampled uniformly in r, theta, phi (HARM stores logarithmic r and
    cos-theta; this stub assumes linear for simplicity).
    """
    spin: float
    r: np.ndarray            # 1D, shape (N_r,)
    theta: np.ndarray        # 1D, shape (N_theta,)
    phi: np.ndarray          # 1D, shape (N_phi,)
    rho: np.ndarray          # gas density, shape (N_r, N_theta, N_phi)
    T_e: np.ndarray          # electron temperature in K
    u_con: np.ndarray        # 4-velocity contravariant, shape (4, N_r, N_theta, N_phi)
    B_con: np.ndarray        # magnetic 4-vector contravariant, shape (4, ...)
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.rho.shape


# ---------------------------------------------------------------------------
# HDF5 reader (requires h5py at call time)
# ---------------------------------------------------------------------------
def load_iharm_snapshot(path: str | Path) -> IHARMFluidSnapshot:
    """Read an iharm3d/HARM HDF5 snapshot.

    The implementation here is intentionally minimal. It assumes the
    PATOKA conventions (header / prims) used by the Illinois group.
    For a different code base (BHAC, Athena++ GR, KORAL) write a small
    adapter to this dataclass.

    Raises ``RuntimeError`` if h5py is missing or the file is not in the
    expected format.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for GRMHD ingestion. Install it via pip.") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GRMHD snapshot not found: {p}")

    with h5py.File(p, "r") as f:
        header = f.get("header") or f.get("Header") or {}
        spin = float(header.get("a", header.get("bhspin", 0.0)) if header else 0.0)
        if "prims" in f:
            prims = f["prims"][...]
        else:
            raise RuntimeError(f"Unsupported HDF5 layout in {p}")

        r = np.asarray(f.get("r", np.linspace(2.0, 50.0, prims.shape[1]))).astype(float)
        theta = np.asarray(f.get("theta", np.linspace(0.01, np.pi - 0.01, prims.shape[2]))).astype(float)
        phi = np.asarray(f.get("phi", np.linspace(0.0, 2 * np.pi, prims.shape[3]))).astype(float)
        # Conventional iharm primitive layout: 0=rho, 1=u, 2=u1,3=u2,4=u3, 5..7=B
        rho = prims[0]
        u_internal = prims[1]
        u_con = np.zeros((4,) + rho.shape, dtype=float)
        u_con[1:] = prims[2:5]
        # u^t solved from normalization (omitted in stub; downstream user reconstructs)
        B_con = np.zeros((4,) + rho.shape, dtype=float)
        B_con[1:] = prims[5:8]
        # Crude electron-temperature proxy: T_e = u_internal / rho
        T_e = u_internal / np.maximum(rho, 1.0e-12) * 1.0e10  # arbitrary units

    return IHARMFluidSnapshot(
        spin=spin,
        r=r, theta=theta, phi=phi,
        rho=rho, T_e=T_e, u_con=u_con, B_con=B_con,
        metadata={"path": str(p)},
    )


# ---------------------------------------------------------------------------
# Synthetic fallback (no GRMHD file needed)
# ---------------------------------------------------------------------------
def synthetic_thin_disk_fluid(
    spin: float = 0.7,
    n_r: int = 64,
    n_theta: int = 32,
    n_phi: int = 32,
    r_max: float = 50.0,
) -> IHARMFluidSnapshot:
    """Build an analytic thin-disk fluid snapshot for testing.

    Density and electron temperature follow Novikov-Thorne radial
    profiles peaked near ISCO. Magnetic field is purely toroidal with
    magnitude ∝ ρ^0.5 (β=10 plasma assumption). Useful as a stand-in
    while real GRMHD snapshots are not yet ingested.
    """
    a = float(np.clip(spin, -0.99, 0.99))
    r_in = isco_radius(a)
    r = np.linspace(r_in + 0.1, r_max, n_r)
    theta = np.linspace(np.pi / 2.0 - 0.05, np.pi / 2.0 + 0.05, n_theta)  # thin slab
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)

    R, TH, PH = np.meshgrid(r, theta, phi, indexing="ij")
    z = R * np.cos(TH)
    H_disc = 0.05 * R  # disc scale height
    rho = (R ** -1.5) * np.exp(-(z / H_disc) ** 2)
    rho /= rho.max()
    rho *= 1.0e-3  # arbitrary scale

    T_max = 1.0e7  # K (X-ray binary thin-disk inner edge)
    T_e = T_max * (R / r_in) ** -0.75 * (rho > 1.0e-6)

    # Keplerian 4-velocity (only u^t, u^φ non-zero)
    omega = 1.0 / (R ** 1.5 + a)
    u_con = np.zeros((4,) + R.shape, dtype=float)
    # u_t solved from g_{tt} + 2 Ω g_{tφ} + Ω² g_{φφ} = -1 (skipped; left zero in stub)
    u_con[3] = omega
    # Toroidal B: B^φ ∝ √ρ
    B_con = np.zeros((4,) + R.shape, dtype=float)
    B_con[3] = np.sqrt(np.maximum(rho, 0.0))

    return IHARMFluidSnapshot(
        spin=spin,
        r=r, theta=theta, phi=phi,
        rho=rho, T_e=T_e, u_con=u_con, B_con=B_con,
        metadata={"source": "synthetic_thin_disk_fluid", "r_in": float(r_in)},
    )


# ---------------------------------------------------------------------------
# Emission coefficient
# ---------------------------------------------------------------------------
def emission_coefficient_thermal(
    rho: np.ndarray,
    T_e: np.ndarray,
    nu_obs_hz: float = 230.0e9,
) -> np.ndarray:
    """Crude thermal-synchrotron emission coefficient (Pacholczyk 1970).

    j_ν ∝ ρ T_e^{1/2} for thermal synchrotron near peak frequency,
    suppressed by exp(-ν / ν_critical). For 230 GHz EHT band,
    ν_critical ≈ ν_thermal_peak (handwave). This is **not** rigorous;
    use only as a placeholder for ray-tracing pipeline integration.
    """
    nu_critical_hz = 1.0e11 * (T_e / 1.0e10)  # arbitrary scaling
    suppress = np.exp(-nu_obs_hz / np.maximum(nu_critical_hz, 1.0e6))
    return rho * np.sqrt(np.maximum(T_e, 1.0e-12)) * suppress
