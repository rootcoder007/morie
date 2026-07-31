"""spstcv -- separable spatio-temporal covariance, Schabenberger & Gotway Sec. 9.2."""

import numpy as np
import pytest

from morie.fn.spstcv import schabenberger_st_cov_separable

CS = lambda h: 2.0 * np.exp(-h / 3.0)          # noqa: E731  sill 2, range 3
CT = lambda k: 1.5 * np.exp(-k / 2.0)          # noqa: E731  sill 1.5, range 2

H = np.array([0.0, 1.0, 2.0, 5.0])
K = np.array([0.0, 1.0, 2.0, 4.0])


def _design(n=30, seed=7):
    rs = np.random.RandomState(seed)
    return rs.uniform(0, 10, size=(n, 2)), rs.uniform(0, 5, n)


def test_product_form_is_the_product():
    r = schabenberger_st_cov_separable(H, K, CS, CT, form="product")
    assert np.allclose(r["st_covariance"], CS(H) * CT(K))


def test_sum_form_is_the_sum():
    """Guards the sum against silently becoming the product."""
    r = schabenberger_st_cov_separable(H, K, CS, CT, form="sum")
    assert np.allclose(r["st_covariance"], CS(H) + CT(K))
    assert not np.allclose(r["st_covariance"], CS(H) * CT(K))


def test_product_sum_form_and_non_separability():
    """De Cesare, Myers and Posa (2001): the text calls it generally nonseparable."""
    r = schabenberger_st_cov_separable(H, K, CS, CT, form="product_sum")
    assert np.allclose(r["st_covariance"], CS(H) * CT(K) + CS(H) + CT(K))
    assert r["separable"] is False


def test_sill_is_c_zero_zero():
    r = schabenberger_st_cov_separable(H, K, CS, CT)
    assert r["sill"] == pytest.approx(CS(0.0) * CT(0.0))
    assert r["sill"] == pytest.approx(3.0)


def test_spatial_shape_is_proportional_across_time_lags():
    """The drawback Sec. 9.2 identifies: no space-time interaction."""
    hh = np.linspace(0.1, 6.0, 25)
    a = schabenberger_st_cov_separable(hh, 0.5, CS, CT)["st_covariance"]
    b = schabenberger_st_cov_separable(hh, 3.0, CS, CT)["st_covariance"]
    ratio = a / b
    assert np.ptp(ratio) < 1e-12


def test_validity_is_checked_on_the_design():
    """eq (9.5) -- construction alone is not proof."""
    coords, times = _design()
    r = schabenberger_st_cov_separable(H, K, CS, CT, coords=coords, times=times)
    assert r["valid"] is True
    assert r["min_eigenvalue"] > -1e-8


def test_c_h_zero_and_c_zero_k_are_the_marginal_functions():
    r = schabenberger_st_cov_separable(H, K, CS, CT)
    assert np.allclose(r["spatial_only"], CS(H) * CT(0.0))
    assert np.allclose(r["temporal_only"], CS(0.0) * CT(K))


def test_negative_spatial_lag_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        schabenberger_st_cov_separable([-1.0], [0.0], CS, CT)


def test_unknown_form_rejected():
    with pytest.raises(ValueError, match="product"):
        schabenberger_st_cov_separable(H, K, CS, CT, form="nonsense")
