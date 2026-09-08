"""Tests for Ghosal Ch 9 rate-example modules."""
import math

from morie.fn.gh_c9_1 import ghosal_logspline_crt
from morie.fn.gh_c9_2 import ghosal_dp_disc_crt
from morie.fn.gh_c9_3 import ghosal_bpoly_crt
from morie.fn.gh_c9_4 import ghosal_dpm_norm_crt
from morie.fn.gh_c9_5 import ghosal_norm_mix_apx
from morie.fn.gh_c9_6 import ghosal_wishart_dpm
from morie.fn.gh_c9_7 import ghosal_whittle_crt
from morie.fn.gh_c9_8 import ghosal_nlar_crt
from morie.fn.gh_c9_9 import ghosal_wn_conj_crt
from morie.fn.gh_c9_10 import ghosal_spline_crt
from morie.fn.gh_c9_11 import ghosal_icens_dp_crt
from morie.fn.gh_sobol_prior import ghosal_sobolev_prior
from morie.fn.gh_wn_gauss_pr import ghosal_white_noise_gauss_prior


def test_logspline_normalized_and_sized():
    r = ghosal_logspline_crt()
    assert r["normalization_gap"] < 1e-9
    assert r["K_n"] == max(1, int(round(500 ** (1.0 / 5.0))))
    assert r["estimate"] > 1.0            # Beta(2,2)-ish peak at 1/2


def test_dp_cdf_half_rate():
    r = ghosal_dp_disc_crt()
    assert r["near_half"] is True


def test_bernstein_improves():
    r = ghosal_bpoly_crt()
    assert r["improving"] is True


def test_dpm_norm_improves():
    r = ghosal_dpm_norm_crt()
    assert r["improving"] is True


def test_mixture_approx_bias_shrinks():
    r = ghosal_norm_mix_apx()
    assert r["improving"] is True
    assert r["estimate"] < 0.2


def test_wishart_dpm_density():
    r = ghosal_wishart_dpm()
    assert r["estimate"] > 0
    assert abs(r["total_mass"] - 1.0) < 0.05


def test_whittle_improves():
    r = ghosal_whittle_crt()
    assert r["improving"] is True


def test_nlar_improves():
    r = ghosal_nlar_crt()
    assert r["improving"] is True


def test_wn_conjugate_exact():
    r = ghosal_wn_conj_crt([1.0, 2.0], 100, [1.0, 0.01])
    # lambda=1: mean = 100/(101), var = 1/101
    assert abs(r["posterior_mean"][0] - 100.0 / 101.0) < 1e-12
    assert abs(r["posterior_var"][0] - 1.0 / 101.0) < 1e-12
    # lambda=0.01: 1/lambda = 100 -> mean = 2*100/200 = 1.0
    assert abs(r["posterior_mean"][1] - 1.0) < 1e-12


def test_spline_rate_near_theory():
    r = ghosal_spline_crt()
    assert abs(r["estimate"] - r["expected_exponent"]) < 0.35


def test_interval_censoring_improves():
    r = ghosal_icens_dp_crt()
    assert r["improving"] is True


def test_sobolev_prior_norms():
    r = ghosal_sobolev_prior()
    assert r["finite_below_s"] is True
    assert r["divergent_at_s_partial"] > 7.0   # harmonic partial sum


def test_wn_gauss_posterior_shrinks_small_prior():
    r = ghosal_white_noise_gauss_prior([1.0, 1.0], 100, [10.0, 0.01])
    assert r["posterior_mean"][0] > 0.99       # wide prior: keep data
    assert r["posterior_mean"][1] < 0.05       # tight prior: shrink
