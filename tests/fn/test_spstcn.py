"""spstcn -- non-separable spatio-temporal covariance, Schabenberger & Gotway Sec. 9.3."""

import math

import numpy as np
import pytest

from morie.fn.spstcn import schabenberger_st_cov_nonsep

HH = np.linspace(0.1, 6.0, 20)


def _design(n=30, seed=7):
    rs = np.random.RandomState(seed)
    return rs.uniform(0, 10, size=(n, 2)), rs.uniform(0, 5, n)


def _cov(h, k, **params):
    return schabenberger_st_cov_nonsep(h, k, params=params,
                                       method="monotone")["st_covariance"]


# --- Sec. 9.3.1, Gneiting ---------------------------------------------------

def test_gneiting_at_zero_lag_is_sigma2():
    """g(0) = psi(0) = 1, so C(0,0) = sigma^2."""
    assert _cov(0.0, 0.0, sigma2=2.5)[()] == pytest.approx(2.5)


def test_gneiting_at_zero_distance_reduces_to_temporal_covariance():
    k = np.array([0.0, 1.0, 2.0, 4.0])
    got = _cov(0.0, k, sigma2=2.0, a=1.0, alpha=1.0, beta=0.6, d=2)
    assert np.allclose(got, 2.0 / (k**2 + 1.0) ** 0.6)


def test_beta_zero_is_separable_and_time_free():
    """(9.9) is separable at beta = 0 and non-separable otherwise."""
    r = schabenberger_st_cov_nonsep(HH, 9.0, params=dict(beta=0.0),
                                    method="monotone")
    assert r["separable"] is True
    assert np.allclose(r["st_covariance"], _cov(HH, 0.0, beta=0.0))
    assert np.allclose(r["st_covariance"], np.exp(-HH**2))


def test_equation_9_9_with_beta_t_zero_equals_equation_9_8():
    a = schabenberger_st_cov_nonsep(HH, 1.3, method="monotone",
                                    params=dict(beta=0.5, beta_t=0.0))
    b = schabenberger_st_cov_nonsep(HH, 1.3, method="monotone",
                                    params=dict(beta=0.5))
    assert a["equation"] == "9.9" and b["equation"] == "9.8"
    assert np.allclose(a["st_covariance"], b["st_covariance"])


@pytest.mark.parametrize("bad", [dict(gamma=1.5), dict(alpha=0.0),
                                 dict(beta=1.5), dict(a=-1.0), dict(c=0.0)])
def test_gneiting_parameter_bounds_enforced(bad):
    """Gneiting's conditions are sufficient for validity; outside them nothing holds."""
    with pytest.raises(ValueError):
        _cov(1.0, 1.0, **bad)


def test_gneiting_is_positive_definite_on_a_design():
    coords, times = _design()
    r = schabenberger_st_cov_nonsep(
        HH, 1.0, method="monotone", coords=coords, times=times,
        params=dict(sigma2=1.0, a=0.5, c=0.3, beta=0.8))
    assert r["valid"] is True


def test_separability_test_halves_the_naive_p_value():
    """Sec. 6.2.3: divide the chi^2 p-value by 2 on the boundary."""
    r = schabenberger_st_cov_nonsep(
        1.0, 1.0, method="monotone",
        params=dict(beta=0.5, neg2_loglik=100.0, neg2_loglik_separable=103.0))
    t = r["separability_test"]
    assert t["p_value"] == pytest.approx(0.5 * t["p_value_naive_chi2_1"])
    assert t["statistic"] == pytest.approx(3.0)


# --- Sec. 9.3.3, Ma's mixtures ---------------------------------------------

def test_power_mixture_poisson_matches_the_explicit_series():
    """eq (9.14) is the pgf of the mixing law at w = Rs Rt."""
    lam, rs_v, rt_v = 1.7, 0.6, 0.4
    r = schabenberger_st_cov_nonsep(
        None, None, method="power_mixture",
        params=dict(rs=rs_v, rt=rt_v, distribution="poisson", lam=lam))
    i = np.arange(80)
    pi = np.exp(-lam) * lam**i / np.array([float(math.factorial(int(m))) for m in i])
    assert float(r["st_covariance"]) == pytest.approx(
        float(np.sum((rs_v * rt_v) ** i * pi)), abs=1e-12)


def test_power_mixture_binomial_pgf():
    r = schabenberger_st_cov_nonsep(
        None, None, method="power_mixture",
        params=dict(rs=0.6, rt=0.4, distribution="binomial", n=4, pi=0.3))
    assert float(r["st_covariance"]) == pytest.approx((0.3 * (0.24 - 1) + 1) ** 4)


def test_bivariate_power_mixture_degenerate_mass():
    """eq (9.13) with all mass at (i, j) = (2, 3) gives Rs^2 Rt^3."""
    pmf = np.zeros((4, 5))
    pmf[2, 3] = 1.0
    r = schabenberger_st_cov_nonsep(None, None, method="power_mixture",
                                    params=dict(rs=0.6, rt=0.4, pmf=pmf))
    assert float(r["st_covariance"]) == pytest.approx(0.6**2 * 0.4**3)


def test_scale_mixture_degenerate_at_one_is_the_product_model():
    """eq (9.16) with F degenerate at u = 1 collapses to Cs(h) Ct(k)."""
    cs = lambda h: 2.0 * np.exp(-h / 3.0)      # noqa: E731
    ct = lambda k: 1.5 * np.exp(-k / 2.0)      # noqa: E731
    r = schabenberger_st_cov_nonsep(
        HH, 1.0, method="scale_mixture",
        params=dict(cov_spatial=cs, cov_temporal=ct, nodes=[1.0], weights=[1.0]))
    assert np.allclose(r["st_covariance"], cs(HH) * ct(1.0))


# --- Sec. 9.3.4, Jones and Zhang -------------------------------------------

def test_smoothness_constraint_enforced():
    """p must exceed max{1, d/2}, else eq (9.17) does not converge."""
    with pytest.raises(ValueError, match="max"):
        schabenberger_st_cov_nonsep(1.0, 1.0, method="differential",
                                    params=dict(p=1.0))


def test_differential_decreases_in_both_lags_and_reports_its_quadrature():
    r = schabenberger_st_cov_nonsep(np.array([0.5, 1.0, 2.0]),
                                    np.array([0.0, 0.0, 0.0]),
                                    method="differential", params=dict(p=2.0))
    c = np.asarray(r["st_covariance"])
    assert np.all(c > 0) and np.all(np.diff(c) < 0)
    q = r["quadrature"]
    assert q["last_panel_rel"] < 1e-9
    assert np.isfinite(q["tail_bound"])


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="method"):
        schabenberger_st_cov_nonsep(1.0, 1.0, method="spectral")
