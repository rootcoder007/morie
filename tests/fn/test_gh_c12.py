"""Tests for Ghosal Ch 12 BvM modules."""
import math

from morie.fn.gh_c12_1 import ghosal_infdim_bvm
from morie.fn.gh_c12_2 import ghosal_dp_bvm
from morie.fn.gh_c12_3 import ghosal_strong_apx_dp
from morie.fn.gh_c12_4 import ghosal_semipara_bvm
from morie.fn.gh_c12_6 import ghosal_semipara_eff
from morie.fn.gh_mises_eff import ghosal_mises_efficiency
from morie.fn.gh_c12_5 import ghosal_eff_infl_fn
from morie.fn.gh_c12_7 import ghosal_strict_sbvm
from morie.fn.gh_c12_8 import ghosal_cox_bvm_sp
from morie.fn.gh_c12_9 import ghosal_wn_full_bvm
from morie.fn.gh_c12_10 import ghosal_wn_lin_bvm
from morie.fn.gh_c12_11 import ghosal_cred_set_cov
from morie.fn.gh_inf_dim_cr import ghosal_inf_dim_credible


def test_parametric_bvm_tv_small():
    assert ghosal_infdim_bvm()["bvm_holds"] is True


def test_dp_bvm_bridge_variance():
    r = ghosal_dp_bvm()
    assert r["gap"] < 0.05


def test_strong_approximation_ks_scale():
    assert ghosal_strong_apx_dp()["typical_ks_range"] is True


def test_semipara_bvm_efficient_variance():
    r = ghosal_semipara_bvm()
    assert r["gap"] < 0.02


def test_efficiency_bound_quadratic():
    # I = diag(2, 4), grad = (1, 2): bound = 1/2 + 4/4 = 1.5
    r = ghosal_semipara_eff([1.0, 2.0], [[2.0, 0.0], [0.0, 4.0]])
    assert abs(r["estimate"] - 1.5) < 1e-9


def test_mises_expansion_exact_for_linear():
    r = ghosal_mises_efficiency()
    assert r["expansion_exact"] is True
    assert abs(r["influence_at_02"] - 0.5) < 1e-12


def test_influence_function_variance():
    data = [0.1, 0.3, 0.6, 0.9]
    r = ghosal_eff_infl_fn(data, 0.5)
    assert r["mean_zero_gap"] < 1e-12
    assert r["matches_bernoulli_var"] is True


def test_strict_sbvm_aggregation():
    assert ghosal_strict_sbvm()["bvm_holds"] is True
    assert ghosal_strict_sbvm(lan_remainder=0.5)["bvm_holds"] \
        is False


def test_cox_partial_likelihood_recovers():
    r = ghosal_cox_bvm_sp()
    assert r["error"] < 0.25


def test_wn_full_bvm_exact():
    r = ghosal_wn_full_bvm()
    assert r["mean_matches_Y"] is True
    assert r["var_matches_In"] is True


def test_linear_functional_variance():
    r = ghosal_wn_lin_bvm()
    assert r["gap"] < 0.25          # ||L||^2 = 1.0


def test_credible_coverage_near_nominal():
    r = ghosal_cred_set_cov()
    assert r["gap"] < 0.06


def test_inf_dim_credible_ball():
    r = ghosal_inf_dim_credible()
    assert r["conservative_or_close"] is True
