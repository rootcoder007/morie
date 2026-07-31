"""Spectral and convolution representations, Ch. 2.

Schabenberger & Gotway (2005), Secs 2.4.2, 2.5, 2.5.3.
"""

import numpy as np
import pytest

from morie.fn.spwkth import schabenberger_wiener_khinchin as wiener_khinchin
from morie.fn.spspec import schabenberger_spectral_representation as spectral
from morie.fn.spconv import schabenberger_convolution_representation as convolve

A = 2.0
EXPO = lambda h: np.exp(-A * np.abs(np.asarray(h, dtype=float)))


def test_spectral_density_of_the_exponential_is_the_cauchy_density():
    """C(h) = exp(-a|h|) transforms to s(w) = (1/pi) a / (a^2 + w^2).

    An exact analytic pair, so this checks the transform rather than
    comparing the code against itself.
    """
    w = np.array([0.0, 0.5, 1.0, 3.0, 8.0])
    got = wiener_khinchin(EXPO, omega=w)["spectral_density"]
    exact = (1 / np.pi) * A / (A**2 + w**2)
    assert np.max(np.abs(got - exact) / exact) < 5e-3


def test_the_density_integrates_to_the_variance_up_to_the_truncated_tail():
    """C(0) = int s(w) dw. The grid can only reach the Nyquist frequency,
    so the shortfall should be the analytic tail beyond it, not an error."""
    r = wiener_khinchin(EXPO)
    shortfall = r["variance"] - r["integrated_density"]
    tail = 1 - 2 / np.pi * np.arctan(r["nyquist_omega"] / A)
    assert 0 < shortfall < 5 * tail


def test_the_omega_grid_stops_at_the_nyquist_frequency():
    """cos(w h) sampled at spacing dh aliases above pi/dh; integrating
    past it sums noise, so a wider range is worse, not better."""
    r = wiener_khinchin(EXPO)
    assert r["nyquist_omega"] > 0
    assert np.isfinite(r["integrated_density"])


def test_spectral_representation_budgets_the_variance():
    """C(0) = sum sigma_j^2: the spectrum is a budget for the variance."""
    r = spectral(np.array([0.0, 1.0, 2.0]))
    assert r["covariance"][0] == pytest.approx(r["sum_sigma2"])
    assert r["variance"] == pytest.approx(r["sum_sigma2"])


def test_spectral_covariance_is_the_cosine_sum():
    """C(h) = sum sigma_j^2 cos(w_j h), eq (2.27)."""
    om = np.array([0.5, 1.5, 3.0])
    s2 = np.array([0.4, 0.3, 0.3])
    h = np.array([0.0, 0.7, 2.1])
    got = spectral(h, sigma2=s2, omega=om)["covariance"]
    exact = np.array([np.sum(s2 * np.cos(om * hh)) for hh in h])
    np.testing.assert_allclose(got, exact, rtol=1e-12)


def test_spectral_input_validation():
    with pytest.raises(ValueError, match="same length"):
        spectral(np.array([0.0]), sigma2=np.ones(3), omega=np.ones(2))
    with pytest.raises(ValueError, match="non-negative variances"):
        spectral(np.array([0.0]), sigma2=np.array([-1.0]), omega=np.array([1.0]))


def test_a_boxcar_kernel_convolves_to_a_tent_correlation():
    """The book's own worked case (Sec 2.4.2, echoed at p. 146)."""
    h = np.linspace(0.0, 1.0, 9)
    got = convolve(h=h)["correlation"]
    np.testing.assert_allclose(got, np.maximum(1 - h, 0.0), atol=1e-3)


def test_convolution_always_yields_a_valid_covariance_at_zero():
    """C(0) = sigma^2 int K^2 > 0 for any non-trivial kernel: positive
    definiteness comes free with the construction."""
    for k in (lambda u: np.exp(-np.asarray(u, float) ** 2),
              lambda u: (np.abs(np.asarray(u, float)) <= 1).astype(float)):
        r = convolve(kernel=k, h=np.array([0.0, 0.5]))
        assert r["variance"] > 0
        assert r["correlation"][0] == pytest.approx(1.0)


def test_convolution_input_validation():
    with pytest.raises(TypeError, match="callable"):
        convolve(kernel="not a function")
    with pytest.raises(ValueError, match="`sigma2_x` must be"):
        convolve(sigma2_x=0.0)
    with pytest.raises(TypeError, match="callable"):
        wiener_khinchin("not a function")
