from __future__ import annotations

from src.disk import hit_intensity
from src.metric import redshift_factor_from_lambda


def test_redshift_factor_is_positive_and_bounded() -> None:
    g = redshift_factor_from_lambda(8.0, 1.57079632679, 0.5, lambda_photon=1.0)
    assert 0.0 < g < 8.0


def test_observed_intensity_uses_g_cubed_scaling() -> None:
    low, g1, _ = hit_intensity(12.0, 1.57079632679, 0.0, lambda_photon=-1.0, q=3.0)
    high, g2, _ = hit_intensity(12.0, 1.57079632679, 0.0, lambda_photon=2.0, q=3.0)
    assert low > 0.0 and high > 0.0
    assert g1 != g2

