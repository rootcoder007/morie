"""Book-certified tests for the Schabenberger & Gotway variogram family.

Every assertion is an identity stated in the book, not a value observed
from running this code. Sources are given per test.

Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
Spatial Data Analysis*. Chapman & Hall/CRC.
"""

import numpy as np
import pytest

from morie.fn.spexp import schabenberger_exponential_variogram as exponential
from morie.fn.spgaus import schabenberger_gaussian_variogram as gaussian
from morie.fn.spsph import schabenberger_spherical_variogram as spherical
from morie.fn.sppow import schabenberger_power_variogram as power
from morie.fn.spnug import schabenberger_nugget_effect as nugget_effect
from morie.fn.spnest import schabenberger_nested_variogram as nested
from morie.fn.spssoc import schabenberger_stationary_cov_semivario as cov_semivario
from morie.fn.spsemv import schabenberger_semivariogram_def as empirical

H = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0])


@pytest.mark.parametrize("model", [exponential, gaussian, spherical])
def test_nugget_is_a_discontinuity_at_the_origin(model):
    """gamma(0) = 0 by definition, even with c0 > 0 (Sec. 4.3.6, p. 150)."""
    g = model(H, nugget=0.4, sill=1.0, range=1.0)["gamma"]
    assert g[0] == 0.0
    # but the limit from the right is the nugget
    tiny = model(np.array([1e-12]), nugget=0.4, sill=1.0, range=1.0)["gamma"]
    assert tiny[0] == pytest.approx(0.4, abs=1e-6)


@pytest.mark.parametrize("model", [exponential, gaussian])
def test_practical_range_is_where_correlation_hits_exp_minus_3(model):
    """alpha is the PRACTICAL range: R(alpha) = exp(-3) (eqs 4.10-4.11)."""
    alpha, sill = 2.5, 3.0
    g = model(np.array([alpha]), nugget=0.0, sill=sill, range=alpha)["gamma"]
    assert g[0] / sill == pytest.approx(1.0 - np.exp(-3.0), rel=1e-12)


def test_spherical_has_a_true_range_not_a_practical_one():
    """Correlation is exactly 0 at h = alpha and stays there (eq 4.13, p. 146)."""
    alpha, sill = 1.7, 2.0
    g = spherical(np.array([alpha, 10 * alpha]), 0.0, sill, alpha)["gamma"]
    assert g[0] == pytest.approx(sill, rel=1e-12)
    assert g[1] == pytest.approx(sill, rel=1e-12)


def test_spherical_matches_the_printed_polynomial():
    """gamma(h) = sigma^2 (3h/2a - (h/a)^3 / 2), h <= a (eq 4.15, p. 147)."""
    alpha, sill = 2.0, 1.5
    h = np.array([0.5, 1.0, 1.9])
    u = h / alpha
    expected = sill * (1.5 * u - 0.5 * u**3)
    got = spherical(h, 0.0, sill, alpha)["gamma"]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_power_model_is_linear_at_lambda_one():
    """lambda = 1 gives the linear semivariogram (Sec. 4.3.5, p. 149)."""
    g = power(np.array([1.0, 2.0, 3.0]), 0.0, 2.0, 1.0)["gamma"]
    np.testing.assert_allclose(g, [2.0, 4.0, 6.0], rtol=1e-12)


def test_power_model_rejects_lambda_at_or_above_two():
    """lambda >= 2 violates the intrinsic hypothesis (p. 149)."""
    with pytest.raises(ValueError, match="intrinsic hypothesis"):
        power(H, 0.0, 1.0, 2.0)


def test_power_model_has_no_sill():
    """It is not second-order stationary: gamma grows without bound."""
    g = power(np.array([1.0, 10.0, 100.0]), 0.0, 1.0, 0.5)["gamma"]
    assert g[0] < g[1] < g[2]


def test_nesting_a_white_noise_component_reproduces_the_nugget_form():
    """gamma_z(h) = sum_j a_j^2 gamma_j(h) (eq 4.23, p. 150).

    A pure-nugget component nested with one continuous model must equal
    that model written with the nugget directly -- the book's own
    justification for the nugget being a nested white-noise term.
    """
    a = nested(H, [{"model": "nugget", "sill": 0.3},
                   {"model": "exponential", "sill": 1.0, "range": 1.0}])["gamma"]
    b = exponential(H, nugget=0.3, sill=1.0, range=1.0)["gamma"]
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_nested_total_sill_is_the_sum_of_component_sills():
    r = nested(H, [{"model": "exponential", "sill": 0.5, "range": 1.0},
                   {"model": "spherical", "sill": 1.25, "range": 3.0}])
    assert r["total_sill"] == pytest.approx(1.75)


def test_gamma_equals_c0_minus_ch_under_second_order_stationarity():
    """gamma(h) = C(0) - C(h); must reproduce the exponential model exactly."""
    sill, alpha = 1.0, 1.0

    def C(h):
        return sill * np.exp(-3.0 * np.asarray(h, dtype=float) / alpha)

    got = cov_semivario(C, H)
    np.testing.assert_allclose(
        got["gamma"], exponential(H, 0.0, sill, alpha)["gamma"], rtol=1e-12
    )
    assert got["sill"] == pytest.approx(sill)


def test_nugget_effect_reports_both_sides_of_the_jump():
    r = nugget_effect(H, nugget=0.25, sill=1.0, range=1.0)
    assert r["gamma_at_zero"] == 0.0
    assert r["limit_at_zero_plus"] == pytest.approx(0.25)
    assert r["total_sill"] == pytest.approx(1.25)


def test_empirical_semivariogram_recovers_the_variance_of_white_noise():
    """A pure-nugget field has gamma(h) = sigma^2 for every h > 0."""
    rng = np.random.default_rng(0)
    coords = rng.random((600, 2))
    sigma = 2.0
    z = rng.normal(0.0, sigma, 600)
    g = empirical(coords, z, n_bins=6)["gamma"]
    assert np.nanmean(g) == pytest.approx(sigma**2, rel=0.15)


def test_empirical_semivariogram_rises_for_a_smooth_field():
    """A strong linear trend gives a monotone-ish rise, not a flat line."""
    rng = np.random.default_rng(1)
    coords = rng.random((400, 2))
    z = 5.0 * coords[:, 0] + rng.normal(0, 0.05, 400)
    r = empirical(coords, z, n_bins=6)
    lag, gam = r["lag"], r["gamma"]
    ok = ~np.isnan(gam)
    assert np.corrcoef(lag[ok], gam[ok])[0, 1] > 0.9


def test_lag_distances_must_be_non_negative():
    for f in (exponential, gaussian, spherical):
        with pytest.raises(ValueError, match="non-negative"):
            f(np.array([-1.0]), 0.0, 1.0, 1.0)


def test_range_must_be_positive():
    with pytest.raises(ValueError, match="`range` must be"):
        exponential(H, 0.0, 1.0, 0.0)
