"""Tests for spnscl.schabenberger_neyman_scott (Thomas-type K-function)."""

import math

import pytest

from morie.fn.spnscl import schabenberger_neyman_scott


def _k_from_lambda2(r, rho, mu, sigma, m=20000):
    """K(r) = (1/lambda^2) integral over |h| <= r of lambda_2(h) dh with
    lambda = rho mu and lambda_2(h) = rho^2 mu^2 + rho mu^2 f(h), f the
    density of the DIFFERENCE of two offspring displacements (bivariate
    normal, variance 2 sigma^2 per axis).  Polar coordinates and a
    composite Simpson rule on [0, r]."""
    lam = rho * mu

    def integrand(t):
        f = math.exp(-t * t / (4 * sigma * sigma)) / (4 * math.pi * sigma * sigma)
        return 2 * math.pi * t * (lam * lam + rho * mu * mu * f)
    d = r / m
    s = integrand(0.0) + integrand(r) + sum((4 if i % 2 else 2) * integrand(i * d) for i in range(1, m))
    return s * d / 3 / (lam * lam)


def test_spnscl_basic():
    """The closed form pi r^2 + (1 - exp(-r^2/(4 sigma^2)))/rho equals the
    integral of the second-order intensity, and lambda = rho mu."""
    rs = [0.05, 0.1, 0.3, 1.0]
    r = schabenberger_neyman_scott(rs, rho=10.0, mu=5.0, sigma=0.1)
    for rr, k in zip(rs, r["k"]):
        assert float(k) == pytest.approx(_k_from_lambda2(rr, 10.0, 5.0, 0.1), rel=1e-10)
    assert r["lambda"] == 50.0
    assert [float(v) for v in r["k_csr"]] == pytest.approx([math.pi * t * t for t in rs], rel=1e-15)
    assert all(float(e) > 0 for e in r["excess"])


def test_spnscl_edge():
    """The excess tends to 1/rho for large r and vanishes at r = 0;
    non-positive parameters and negative distances raise."""
    r = schabenberger_neyman_scott([0.0, 50.0], rho=4.0, sigma=0.5)
    assert float(r["excess"][0]) == 0.0
    assert float(r["excess"][1]) == pytest.approx(0.25, abs=1e-15)
    with pytest.raises(ValueError):
        schabenberger_neyman_scott([1.0], rho=0.0)
    with pytest.raises(ValueError):
        schabenberger_neyman_scott([-1.0])
