"""Tests for spconv.schabenberger_convolution_representation."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.spconv import schabenberger_convolution_representation


def test_spconv_basic():
    """A Gaussian kernel K(u) = exp(-u^2/2) convolves to
    C(h) = sigma2 sqrt(pi) exp(-h^2/4).  The trapezoid rule is
    spectrally accurate for this smooth, rapidly decaying integrand, and
    at half-width 12 the truncated tail is below exp(-72), so the
    quadrature is exact to rounding."""
    h = [0.0, 0.5, 1.0, 2.0, 3.0]
    r = schabenberger_convolution_representation(
        kernel=lambda u: np.exp(-0.5 * np.asarray(u, dtype=float) ** 2),
        h=h, sigma2_x=2.0, half_width=12.0, n=4001)
    for hh, c, rho in zip(h, r["covariance"], r["correlation"]):
        assert float(c) == pytest.approx(2.0 * math.sqrt(math.pi) * math.exp(-hh * hh / 4), rel=1e-12)
        assert float(rho) == pytest.approx(math.exp(-hh * hh / 4), rel=1e-12)
    assert r["variance"] == pytest.approx(2.0 * math.sqrt(math.pi), rel=1e-12)


def test_spconv_edge():
    """The default boxcar gives the tent 1 - |h| on [0, 1] and 0 beyond.
    The integrand is a product of indicators, so each of its two jumps
    costs the trapezoid rule at most one node spacing du = 10/40000;
    the tolerance is 2 du.  A non-callable kernel or sigma2 <= 0 raise."""
    r = schabenberger_convolution_representation()
    du = 10.0 / 40000
    for hh, rho in zip(r["h"], r["correlation"]):
        assert float(rho) == pytest.approx(max(0.0, 1.0 - float(hh)), abs=2 * du)
    with pytest.raises(TypeError):
        schabenberger_convolution_representation(kernel=3.0)
    with pytest.raises(ValueError):
        schabenberger_convolution_representation(sigma2_x=0.0)
