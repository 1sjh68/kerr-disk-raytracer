from __future__ import annotations

import math

import numpy as np

from .metric import isco_radius, redshift_factor_from_lambda


def disk_inner_radius(a: float, value: str | float = "isco") -> float:
    if isinstance(value, str) and value.lower() == "isco":
        return isco_radius(a)
    return float(value)


def emissivity_power_law(r: np.ndarray | float, q: float = 3.0) -> np.ndarray | float:
    return np.maximum(np.asarray(r, dtype=float), 1.0e-6) ** (-q)


def novikov_thorne_flux(r: np.ndarray | float, r_in: float) -> np.ndarray | float:
    rr = np.maximum(np.asarray(r, dtype=float), r_in)
    flux = rr ** -3.0 * np.maximum(0.0, 1.0 - np.sqrt(r_in / rr))
    return flux


def hit_intensity(
    r: float,
    theta: float,
    a: float,
    lambda_photon: float,
    q: float = 3.0,
    model: str = "power_law",
    r_in: float | None = None,
) -> tuple[float, float, float]:
    if r_in is None:
        r_in = isco_radius(a)
    if model == "novikov_thorne":
        emitted = float(novikov_thorne_flux(r, r_in))
    else:
        emitted = float(emissivity_power_law(r, q=q))
    g = redshift_factor_from_lambda(r, theta, a, lambda_photon)
    observed = emitted * g**3
    temperature = max(emitted, 0.0) ** 0.25
    return observed, g, temperature


def temperature_to_rgb(temp: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    """Map normalized disk temperature/redshift to display RGB."""

    t = np.asarray(temp, dtype=float)
    if np.max(t) > 0.0:
        t = t / np.max(t)
    g = np.clip(redshift, 0.2, 2.5)
    hot = np.stack(
        [
            np.ones_like(t),
            np.clip(0.35 + 0.85 * t, 0.0, 1.0),
            np.clip(0.08 + 0.75 * t**1.7, 0.0, 1.0),
        ],
        axis=-1,
    )
    blue = np.stack(
        [
            np.clip(0.55 + 0.45 / g, 0.0, 1.0),
            np.clip(0.65 + 0.30 * t, 0.0, 1.0),
            np.clip(0.75 + 0.25 * g / 2.5, 0.0, 1.0),
        ],
        axis=-1,
    )
    mix = np.clip((g - 0.75) / 0.8, 0.0, 1.0)[..., None]
    return hot * (1.0 - mix) + blue * mix


def temperature_to_rgb_dispatch(
    temp: np.ndarray,
    redshift: np.ndarray,
    *,
    mode: str = "approx",
    cie_t_min: float = 3000.0,
    cie_t_max: float = 30000.0,
) -> np.ndarray:
    """Choose between the legacy approximate map and the spectral CIE 1931 map.

    `mode="approx"` uses the original analytic blend (default for backwards
    compatibility). `mode="cie1931"` invokes the Planck/CIE/sRGB pipeline in
    :mod:`src.disk_color`, which is closer to the perceptually correct colour
    of a black-body emitter at temperature ``cie_t_min..cie_t_max`` Kelvin
    redshifted by the local g-factor.
    """
    mode_norm = (mode or "approx").lower()
    if mode_norm in {"cie", "cie1931", "spectral"}:
        from .disk_color import temperature_to_rgb_cie1931

        return temperature_to_rgb_cie1931(
            temp, redshift, t_min_kelvin=cie_t_min, t_max_kelvin=cie_t_max
        )
    if mode_norm not in {"approx", "approximate", "legacy", "default"}:
        raise ValueError(f"unknown disk colour mode: {mode!r}")
    return temperature_to_rgb(temp, redshift)


def bloom_filter(image: np.ndarray) -> np.ndarray:
    """Small dependency-free separable blur for bright pixels."""

    bright = np.maximum(image - 0.65, 0.0)
    if not np.any(bright):
        return image
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    tmp = bright.copy()
    pad = len(kernel) // 2
    for axis in [0, 1]:
        padded = np.pad(tmp, [(pad, pad) if i == axis else (0, 0) for i in range(tmp.ndim)], mode="edge")
        out = np.zeros_like(tmp)
        for i, weight in enumerate(kernel):
            sl = [slice(None)] * tmp.ndim
            sl[axis] = slice(i, i + tmp.shape[axis])
            out += weight * padded[tuple(sl)]
        tmp = out
    return np.clip(image + 0.35 * tmp, 0.0, None)


def reinhard_tone_map(rgb: np.ndarray, gamma: float = 2.2, bloom: bool = True) -> np.ndarray:
    mapped = rgb / (1.0 + rgb)
    if bloom:
        mapped = bloom_filter(mapped)
    return np.clip(mapped, 0.0, 1.0) ** (1.0 / gamma)


def doppler_factor(r: np.ndarray, phi: np.ndarray, inclination_rad: float, a: float) -> np.ndarray:
    speed = np.sqrt(1.0 / np.maximum(r, 2.0))
    speed *= np.clip(1.0 + 0.25 * a / np.maximum(r, 2.0) ** 1.5, 0.6, 1.4)
    gamma = 1.0 / np.sqrt(np.maximum(1.0e-6, 1.0 - speed * speed))
    return 1.0 / (gamma * np.maximum(0.08, 1.0 - speed * math.sin(inclination_rad) * np.cos(phi)))

