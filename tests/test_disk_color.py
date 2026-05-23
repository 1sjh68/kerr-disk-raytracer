"""Tests for the CIE 1931 / Planck colour pipeline in src/disk_color.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.disk_color import (
    cie1931_xyz_bar,
    cie1931_xyz_from_spectrum,
    linear_to_srgb,
    planck_spectrum,
    temperature_to_rgb_cie1931,
    temperature_to_srgb,
    xyz_to_linear_srgb,
)


def test_planck_peak_wavelength_obeys_wien() -> None:
    """λ_max * T ≈ 2.898e-3 m·K (Wien's displacement law)."""
    wl_nm = np.linspace(100.0, 3000.0, 4000)
    for T in [1500.0, 3000.0, 5800.0, 10000.0]:
        spec = planck_spectrum(wl_nm, T)
        peak_idx = int(np.argmax(spec))
        peak_lambda_m = wl_nm[peak_idx] * 1.0e-9
        product = peak_lambda_m * T
        assert 2.6e-3 < product < 3.2e-3, f"Wien violation T={T}: λT={product:.3e}"


def test_cie_xyz_bar_peaks_match_standard_observer() -> None:
    """x̄ peaks near 600 nm, ȳ near 555 nm, z̄ near 445 nm (CIE 1931)."""
    wl = np.arange(380.0, 781.0, 1.0)
    xyz = cie1931_xyz_bar(wl)
    x_peak = wl[int(np.argmax(xyz[..., 0]))]
    y_peak = wl[int(np.argmax(xyz[..., 1]))]
    z_peak = wl[int(np.argmax(xyz[..., 2]))]
    assert 590 < x_peak < 615, f"x̄ peak {x_peak} nm, expected ~600"
    assert 550 < y_peak < 575, f"ȳ peak {y_peak} nm, expected ~555"
    assert 435 < z_peak < 460, f"z̄ peak {z_peak} nm, expected ~445"


def test_cie_xyz_bar_nonneg() -> None:
    wl = np.arange(380.0, 781.0, 1.0)
    xyz = cie1931_xyz_bar(wl)
    assert (xyz >= -1.0e-6).all()


def test_sun_5778k_is_near_white_chromaticity() -> None:
    """The Sun's surface ~5778 K should land near the achromatic axis.

    Solar chromaticity in CIE 1931 xyY is approximately x≈0.32, y≈0.33.
    """
    rgb = temperature_to_srgb(5778.0)
    # All sRGB channels close to each other → near-white
    spread = float(rgb.max() - rgb.min())
    assert spread < 0.25, f"5778K should look near-white; rgb={rgb}, spread={spread}"


def test_3000k_is_red_dominant() -> None:
    rgb = temperature_to_srgb(3000.0)
    assert rgb[0] > rgb[2], f"3000 K should be red>blue; rgb={rgb}"


def test_15000k_is_blue_dominant() -> None:
    rgb = temperature_to_srgb(15000.0)
    assert rgb[2] > rgb[0], f"15000 K should be blue>red; rgb={rgb}"


def test_temperature_to_srgb_handles_array_shape() -> None:
    T = np.array([[3000.0, 5800.0], [10000.0, 20000.0]])
    rgb = temperature_to_srgb(T)
    assert rgb.shape == T.shape + (3,)
    assert (rgb >= 0).all() and (rgb <= 1.0).all()


def test_xyz_to_linear_srgb_identity_on_white() -> None:
    """D65 white XYZ ≈ (0.95047, 1.0, 1.08883) → equal RGB."""
    xyz = np.array([0.95047, 1.0, 1.08883])
    rgb = xyz_to_linear_srgb(xyz)
    spread = float(rgb.max() - rgb.min())
    assert spread < 0.05, f"D65 white should give equal RGB; rgb={rgb}"


def test_linear_to_srgb_endpoints() -> None:
    assert linear_to_srgb(np.array([0.0]))[0] == pytest.approx(0.0, abs=1.0e-6)
    assert linear_to_srgb(np.array([1.0]))[0] == pytest.approx(1.0, abs=1.0e-6)


def test_temperature_to_rgb_cie1931_disk_pipeline_smoke() -> None:
    """End-to-end: hit-mask false stays black; hot disk pixel non-zero RGB."""
    t = np.zeros((4, 4))
    g = np.zeros((4, 4))
    # one hot pixel
    t[2, 2] = 1.0
    g[2, 2] = 1.0
    rgb = temperature_to_rgb_cie1931(t, g)
    assert rgb.shape == (4, 4, 3)
    assert (rgb[0, 0] == 0).all()
    assert (rgb[2, 2] > 0).any()


def test_temperature_to_rgb_cie1931_redshift_reddens() -> None:
    """A redshifted disk pixel (g < 1) should be redder than a non-shifted one."""
    t = np.array([1.0, 1.0])
    g_blue = np.array([1.0, 1.0])
    g_red = np.array([1.0, 0.4])
    rgb_blue = temperature_to_rgb_cie1931(t, g_blue, t_min_kelvin=8000.0, t_max_kelvin=8000.0)
    rgb_red = temperature_to_rgb_cie1931(t, g_red, t_min_kelvin=8000.0, t_max_kelvin=8000.0)
    # The redshifted (g=0.4) pixel should have higher red/blue ratio
    rb_blue = rgb_blue[1, 0] / max(rgb_blue[1, 2], 1e-9)
    rb_red = rgb_red[1, 0] / max(rgb_red[1, 2], 1e-9)
    assert rb_red > rb_blue, f"redshifted pixel should be redder: rb_blue={rb_blue}, rb_red={rb_red}"


def test_temperature_to_rgb_cie1931_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        temperature_to_rgb_cie1931(np.zeros(4), np.zeros(5))
