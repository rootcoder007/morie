"""Ch. 2 definitional family. Schabenberger & Gotway (2005) Secs 2.3, 2.4."""

import numpy as np
import pytest

from morie.fn.spcont import schabenberger_spatial_continuity as continuity
from morie.fn.spmsd import schabenberger_mean_square_diff as differentiability
from morie.fn.spcovf import schabenberger_covariance_function as covfun

EXPO = lambda h: np.exp(-3 * np.asarray(h, dtype=float))
GAUSS = lambda h: np.exp(-3 * np.asarray(h, dtype=float) ** 2)
SPH = lambda h: np.where(np.asarray(h, float) <= 1,
                         1 - 1.5 * np.asarray(h, float)
                         + 0.5 * np.asarray(h, float) ** 3, 0.0)
NUGGET = lambda h: np.where(np.asarray(h, float) == 0, 1.3, EXPO(h))


@pytest.mark.parametrize("cov", [EXPO, GAUSS, SPH])
def test_continuous_covariances_give_ms_continuity(cov):
    """MS continuous iff C is continuous at the origin (p. 50)."""
    assert continuity(cov)["is_continuous"]


def test_a_nugget_destroys_mean_square_continuity():
    """"A process that exhibits a discontinuity at the origin cannot be
    mean square continuous" (p. 50)."""
    r = continuity(NUGGET)
    assert not r["is_continuous"]
    # The limit is probed at a finite h, so the estimate is the nugget
    # PLUS the continuous part's decay over that lag: at h = 1e-6 the
    # exponential component contributes 3e-6. Demanding 1e-6 here asks
    # for more precision than probing a limit at finite h can give.
    assert r["nugget"] == pytest.approx(0.3, abs=1e-4)


def test_the_decision_is_shrinkage_not_a_fixed_threshold():
    """A continuous C still has a nonzero gap at any finite h; a nugget's
    gap plateaus. The ratio separates them."""
    assert continuity(EXPO)["gap_ratio"] < 0.1     # shrinking
    assert continuity(NUGGET)["gap_ratio"] > 0.5   # plateau


def test_gaussian_covariance_is_mean_square_differentiable():
    """C(h) = exp(-3h^2) is infinitely differentiable (eq 2.6, p. 51)."""
    r = differentiability(GAUSS, m=1)
    assert r["is_differentiable"]
    assert r["derivative_2m"] == pytest.approx(-6.0, rel=1e-3)  # C''(0) = -6
    assert differentiability(GAUSS, m=2)["is_differentiable"]


@pytest.mark.parametrize("cov", [EXPO, SPH])
def test_kinked_covariances_are_not_differentiable(cov):
    """No second derivative at 0, so not MS differentiable (Stein 1999)."""
    r = differentiability(cov, m=1)
    assert not r["is_differentiable"]
    assert r["growth_ratio"] > 1.5      # diverges as the stencil shrinks


def test_derivative_field_covariance_sign():
    """Cov of the m-th derivative field is (-1)^m d^{2m}C/dh^{2m}."""
    r = differentiability(GAUSS, m=1)
    assert r["derivative_cov"] == pytest.approx(-r["derivative_2m"])
    assert r["derivative_cov"] > 0      # a variance


def test_differentiability_input_validation():
    with pytest.raises(TypeError, match="callable"):
        differentiability("not a function")
    with pytest.raises(ValueError, match="`m` must be"):
        differentiability(GAUSS, m=0)
    with pytest.raises(ValueError, match="`h` must be"):
        differentiability(GAUSS, h=0.0)


def test_empirical_covariance_recovers_the_sill():
    rng = np.random.default_rng(0)
    coords = rng.random((400, 2)) * 10
    z = rng.normal(0, 2.0, 400)
    r = covfun(coords, z, n_bins=6)
    assert r["sill"] == pytest.approx(4.0, rel=0.25)


def test_implied_semivariogram_matches_the_direct_estimate():
    """gamma(h) = C(0) - C(h) holds under second-order stationarity; the
    two are computed independently so agreement is a real check."""
    rng = np.random.default_rng(1)
    coords = rng.random((500, 2)) * 10
    z = rng.normal(0, 2.0, 500)
    r = covfun(coords, z, n_bins=6)
    ok = ~np.isnan(r["covariance"]) & ~np.isnan(r["semivariogram"])
    assert np.max(np.abs(r["semivariogram"][ok]
                         - r["implied_semivariogram"][ok])) < 0.5


def test_covariance_input_validation():
    with pytest.raises(ValueError, match="same number of rows"):
        covfun(np.zeros((5, 2)), np.zeros(4))
    with pytest.raises(TypeError, match="callable"):
        continuity("not a function")
