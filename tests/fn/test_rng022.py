"""rng022: rho_xy = C_xy/(sigma_x sigma_y) (Rangayyan 2024, Eq. 3.22, p. 98)."""

import numpy as np
import pytest

from morie.fn.rng022 import rangayyan_ch3_correlation_coefficient as rho


def test_rng022_known_value():
    """C=6, sigma_x=3, sigma_y=4 -> 6/12 = 0.5."""
    assert rho(6.0, 3.0, 4.0)["value"] == 0.5


def test_rng022_matches_numpy_corrcoef_on_real_data():
    """Feed the book's own covariance definition (Eq. 3.21) and cross-check.

    numpy computes the correlation by a different route, so agreement pins the
    normalisation rather than restating it.
    """
    rng = np.random.default_rng(7)
    n = 5000
    x = rng.standard_normal(n)
    y = 0.6 * x + rng.standard_normal(n) * 0.8
    C_xy = float(np.mean((x - x.mean()) * (y - y.mean())))
    got = rho(C_xy, float(np.std(x)), float(np.std(y)))["value"]
    assert got == pytest.approx(float(np.corrcoef(x, y)[0, 1]), abs=1e-12)


def test_rng022_perfect_correlation_hits_the_bound():
    """C = sigma_x*sigma_y is the Cauchy-Schwarz equality case: rho = 1."""
    assert rho(12.0, 3.0, 4.0)["value"] == pytest.approx(1.0)
    assert rho(-12.0, 3.0, 4.0)["value"] == pytest.approx(-1.0)


def test_rng022_uncorrelated_gives_zero():
    assert rho(0.0, 3.0, 4.0)["value"] == 0.0


def test_rng022_rejects_out_of_range_result():
    """The book states -1 <= rho <= +1 as part of the definition.

    Inputs implying |rho| > 1 cannot come from one pair of processes, so they
    raise instead of returning an unusable "correlation" above 1.
    """
    with pytest.raises(ValueError, match=r"outside the \[-1, \+1\] range"):
        rho(13.0, 3.0, 4.0)


def test_rng022_rejects_zero_sd():
    """rho is undefined for a degenerate process -- not zero, not inf."""
    with pytest.raises(ValueError, match="strictly positive"):
        rho(0.0, 0.0, 4.0)
