"""Spectrally correct disk-temperature -> sRGB colour mapping.

This module replaces the ad-hoc `temperature_to_rgb` heuristic in
:mod:`src.disk` for high-fidelity rendering. The pipeline is:

  T (Kelvin) -> Planck spectrum I(λ, T)
             -> integrate against CIE 1931 2-deg standard observer x̄ ȳ z̄
             -> XYZ tristimulus
             -> linear sRGB (D65 white)
             -> gamma-encoded sRGB display values

The CIE 1931 colour-matching functions are produced by the multi-lobe
Gaussian fit of Wyman, Sloan & Shirley (2013), which matches the
tabulated standard observer to better than 0.5% over 380-780 nm and is
the fastest analytic representation suitable for vectorised numpy.

References
----------
Wyman, C., Sloan, P.-P., Shirley, P., "Simple Analytic Approximations
to the CIE XYZ Color Matching Functions", JCGT 2(2), 2013.
http://jcgt.org/published/0002/02/01/

Public API
----------
- planck_spectrum
- cie1931_xyz_from_spectrum
- xyz_to_linear_srgb
- linear_to_srgb
- temperature_to_srgb
- temperature_to_rgb_cie1931
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
PLANCK_H = 6.62607015e-34          # J·s
SPEED_OF_LIGHT = 2.99792458e8      # m/s
BOLTZMANN_K = 1.380649e-23         # J/K

# Default visible wavelength sampling (nm). 5 nm resolution is enough
# for sub-percent accuracy in XYZ integration.
_DEFAULT_LAMBDA_NM = np.arange(380.0, 781.0, 5.0)

# sRGB transfer constants (IEC 61966-2-1:1999)
_SRGB_CUTOFF = 0.0031308

# linear sRGB matrix from CIE XYZ (D65 white point)
_XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# CIE 1931 2-deg colour matching functions (Wyman 2013 multi-lobe fit)
# ---------------------------------------------------------------------------
def _gauss_asym(wavelength_nm: np.ndarray, mu: float, sigma1: float, sigma2: float) -> np.ndarray:
    """Asymmetric Gaussian: sigma1 below the peak, sigma2 above."""
    wl = np.asarray(wavelength_nm, dtype=float)
    sigma = np.where(wl < mu, sigma1, sigma2)
    t = (wl - mu) / sigma
    return np.exp(-0.5 * t * t)


def cie1931_xyz_bar(wavelength_nm: np.ndarray) -> np.ndarray:
    """Return tabulated x̄ ȳ z̄ at given wavelengths (nm).

    Output shape: input.shape + (3,).
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    x_bar = (
        1.056 * _gauss_asym(wl, 599.8, 37.9, 31.0)
        + 0.362 * _gauss_asym(wl, 442.0, 16.0, 26.7)
        - 0.065 * _gauss_asym(wl, 501.1, 20.4, 26.2)
    )
    y_bar = (
        0.821 * _gauss_asym(wl, 568.8, 46.9, 40.5)
        + 0.286 * _gauss_asym(wl, 530.9, 16.3, 31.1)
    )
    z_bar = (
        1.217 * _gauss_asym(wl, 437.0, 11.8, 36.0)
        + 0.681 * _gauss_asym(wl, 459.0, 26.0, 13.8)
    )
    return np.stack([x_bar, y_bar, z_bar], axis=-1)


# ---------------------------------------------------------------------------
# Planck spectrum
# ---------------------------------------------------------------------------
def planck_spectrum(
    wavelength_nm: np.ndarray | float,
    temperature_kelvin: np.ndarray | float,
) -> np.ndarray:
    """Spectral radiance of a black body, Planck's law.

    Returns intensity I(λ, T) in W·sr⁻¹·m⁻³ (per unit wavelength).
    Result shape broadcasts λ and T according to numpy rules.

    For very small T (≤0) returns 0; for large hν/(kT) overflow we
    clamp the exponent to avoid `inf`.
    """
    wl_m = np.asarray(wavelength_nm, dtype=float) * 1.0e-9
    T = np.asarray(temperature_kelvin, dtype=float)

    # Compute hc / (lambda * k * T) safely
    valid_T = np.maximum(T, 1.0e-12)
    x = (PLANCK_H * SPEED_OF_LIGHT) / (wl_m * BOLTZMANN_K * valid_T)
    x = np.minimum(x, 700.0)  # avoid overflow in exp

    numer = 2.0 * PLANCK_H * SPEED_OF_LIGHT * SPEED_OF_LIGHT
    denom = wl_m**5 * (np.exp(x) - 1.0)
    out = numer / denom
    out = np.where(T > 0, out, 0.0)
    return out


# ---------------------------------------------------------------------------
# Spectrum -> CIE XYZ
# ---------------------------------------------------------------------------
def cie1931_xyz_from_spectrum(
    spectrum: np.ndarray,
    wavelength_nm: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate a spectrum against CIE 1931 colour matching functions.

    `spectrum` shape: (..., N_lambda).
    Output shape: (..., 3) (X, Y, Z).
    """
    wl = _DEFAULT_LAMBDA_NM if wavelength_nm is None else np.asarray(wavelength_nm, dtype=float)
    spec = np.asarray(spectrum, dtype=float)
    if spec.shape[-1] != wl.shape[0]:
        raise ValueError(
            f"spectrum last axis {spec.shape[-1]} does not match wavelength count {wl.shape[0]}"
        )
    bar = cie1931_xyz_bar(wl)  # (N_lambda, 3)
    # ∫ spec(λ) bar(λ) dλ via trapezoidal rule
    integrand = spec[..., :, None] * bar[None, ...] if spec.ndim == 1 else spec[..., None] * bar
    if spec.ndim == 1:
        integrand = spec[:, None] * bar
    xyz = np.trapezoid(integrand, x=wl, axis=-2)
    return xyz


# ---------------------------------------------------------------------------
# XYZ -> sRGB
# ---------------------------------------------------------------------------
def xyz_to_linear_srgb(xyz: np.ndarray) -> np.ndarray:
    """Apply the linear D65 XYZ -> sRGB matrix. Negative components clamped to 0."""
    arr = np.asarray(xyz, dtype=float)
    rgb = arr @ _XYZ_TO_LINEAR_SRGB.T
    return np.maximum(rgb, 0.0)


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """Apply IEC 61966-2-1 sRGB gamma transfer."""
    arr = np.asarray(linear_rgb, dtype=float)
    out = np.where(
        arr <= _SRGB_CUTOFF,
        12.92 * arr,
        1.055 * np.power(np.maximum(arr, 1.0e-9), 1.0 / 2.4) - 0.055,
    )
    return np.clip(out, 0.0, 1.0)


def temperature_to_srgb(
    temperature_kelvin: np.ndarray | float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Black-body temperature (K) -> gamma-encoded sRGB triple.

    Output shape matches input.shape + (3,).
    With `normalize=True` (default) each spectrum is divided by its own
    Y luminance so all colours have luminance 1; useful for chromaticity
    visualisation. With `normalize=False` the absolute Planck intensity
    is preserved, so brightness scales with T^4 (Stefan-Boltzmann)
    before tone mapping.
    """
    T = np.asarray(temperature_kelvin, dtype=float)
    wl = _DEFAULT_LAMBDA_NM
    # Build (T.shape + (N_lambda,)) spectrum
    spec = planck_spectrum(wl, T[..., None]) if T.ndim > 0 else planck_spectrum(wl, float(T))
    xyz = cie1931_xyz_from_spectrum(spec, wl)  # (... , 3)
    if normalize:
        Y = np.maximum(xyz[..., 1:2], 1.0e-12)
        xyz = xyz / Y
    rgb_linear = xyz_to_linear_srgb(xyz)
    return linear_to_srgb(rgb_linear)


# ---------------------------------------------------------------------------
# Drop-in API matching disk.temperature_to_rgb signature
# ---------------------------------------------------------------------------
def temperature_to_rgb_cie1931(
    temp_unitless: np.ndarray,
    redshift: np.ndarray,
    t_min_kelvin: float = 3000.0,
    t_max_kelvin: float = 30000.0,
) -> np.ndarray:
    """Spectrally correct CIE 1931 / Planck / sRGB mapping for disk renders.

    The project uses unit-free emitted intensity ``I_emit ~ r^-q`` with
    ``temperature = I_emit ** 0.25``. This routine maps the *normalised*
    unitless field to a physical black-body range
    ``[t_min_kelvin, t_max_kelvin]`` (defaults span 3000-30000 K, the
    visible chromaticity locus from red dwarf to early-type stars), then
    multiplies by the gravitational + Doppler redshift factor ``g``
    (Wien displacement law: a redshifted black body of emitted
    temperature T appears as if it had temperature ``g * T``).

    Pixels with temperature ≤ 0 are returned as black.

    Parameters
    ----------
    temp_unitless : array_like
        The ``temperature`` field from :func:`disk.hit_intensity`; same
        units as in the rest of the project.
    redshift : array_like
        The frequency-shift factor ``g`` from the same map.
    t_min_kelvin, t_max_kelvin : float
        Real-temperature range mapped to the unitless field.

    Returns
    -------
    rgb : ndarray
        Same shape as ``temp_unitless`` plus a trailing 3 axis,
        gamma-encoded sRGB in [0, 1].
    """
    t = np.asarray(temp_unitless, dtype=float)
    g = np.asarray(redshift, dtype=float)
    if t.shape != g.shape:
        raise ValueError(f"temperature and redshift shapes differ: {t.shape} vs {g.shape}")

    nonzero = t > 0
    if not np.any(nonzero):
        return np.zeros(t.shape + (3,), dtype=float)

    # Normalise the unitless temperature to [0, 1] over disk-hit pixels
    t_max = float(t[nonzero].max())
    t_norm = np.zeros_like(t)
    t_norm[nonzero] = t[nonzero] / max(t_max, 1.0e-12)

    # Map to physical Kelvin range
    T_emit = t_min_kelvin + (t_max_kelvin - t_min_kelvin) * t_norm

    # Apply redshift via Wien displacement law (frequency-shift)
    T_obs = T_emit * np.clip(g, 1.0e-3, 5.0)

    rgb = temperature_to_srgb(T_obs, normalize=True)
    rgb = np.where(nonzero[..., None], rgb, 0.0)
    return rgb
