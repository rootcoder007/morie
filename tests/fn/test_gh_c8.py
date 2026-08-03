"""Tests for Ghosal Ch 8 contraction-rate modules."""
import math

from morie.fn.gh_c8_1 import ghosal_crt_def
from morie.fn.gh_c8_2 import ghosal_ggv_thm
from morie.fn.gh_c8_4 import ghosal_prior_mass_cnd
from morie.fn.gh_c8_5 import ghosal_entropy_cnd
from morie.fn.gh_c8_7 import ghosal_fin_apx_pri
from morie.fn.gh_c8_8 import ghosal_gauss_reg_crt
from morie.fn.gh_c8_9 import ghosal_markov_crt
from morie.fn.gh_c8_10 import ghosal_wn_crt
from morie.fn.gh_c8_11 import ghosal_ts_crt
from morie.fn.gh_c8_12 import ghosal_crt_lower
from morie.fn.gh_c8_13 import ghosal_misspec_crt
from morie.fn.gh_c8_14 import ghosal_convex_misp
from morie.fn.gh_c8_15 import ghosal_alpha_pst_crt
from morie.fn.gh_contr_rate2 import ghosal_contraction_rate_iid
from morie.fn.gh_wn_rate_opt import ghosal_white_noise_optimal_rate


def test_crt_def_mass_falls_in_M():
    r = ghosal_crt_def()
    assert r["decreasing_in_M"] is True
    assert r["estimate"] < 0.05


def test_ggv_conditions():
    # n=100, eps=0.2: n eps^2 = 4; feasible spec
    ok = ghosal_ggv_thm(100, 0.2, -3.0, 3.5, -25.0)
    assert ok["rate_certified"] is True
    bad = ghosal_ggv_thm(100, 0.2, -10.0, 3.5, -25.0)
    assert bad["rate_certified"] is False


def test_prior_mass_positive():
    r = ghosal_prior_mass_cnd([0.25] * 4)
    assert r["positive"] is True


def test_entropy_condition():
    ok = ghosal_entropy_cnd(5, 1.0, 10000, 0.1)
    assert ok["condition_holds"] is True     # 5 log30 ~ 17 <= 100
    bad = ghosal_entropy_cnd(500, 1.0, 10000, 0.1)
    assert bad["condition_holds"] is False


def test_net_prior_rate_balance():
    r = ghosal_fin_apx_pri(2.0, 100000)
    assert r["balance_gap"] < 1e-9
    assert abs(r["estimate"] - 100000 ** (-0.4)) < 1e-12


def test_gauss_reg_rate_matches_theory():
    r = ghosal_gauss_reg_crt()
    assert abs(r["estimate"] - r["expected_exponent"]) < 0.12


def test_markov_parametric_rate():
    r = ghosal_markov_crt()
    assert abs(r["estimate"] - 1.0) < 0.25


def test_wn_rate_matches_theory():
    r = ghosal_wn_crt()
    assert abs(r["estimate"] - r["expected_exponent"]) < 0.12


def test_whittle_contracts():
    r = ghosal_ts_crt()
    assert r["contracting"] is True


def test_lower_bound_balance():
    r = ghosal_crt_lower(1.5, 10000)
    assert r["balance_gap"] < 1e-9
    assert abs(r["exponent"] - 0.375) < 1e-12


def test_misspec_targets_projection():
    r = ghosal_misspec_crt()
    assert r["error_to_projection"] < 0.05


def test_convex_kl_unique_min():
    r = ghosal_convex_misp([0.5, 0.5], [0.9, 0.1], [0.1, 0.9])
    assert r["convex_along_segment"] is True
    assert abs(r["estimate"] - 0.5) < 0.02   # symmetric setup


def test_alpha_posterior_same_rate():
    r = ghosal_alpha_pst_crt()
    assert r["parametric_rate"] is True


def test_iid_l1_half_rate():
    r = ghosal_contraction_rate_iid()
    assert r["half_rate"] is True


def test_wn_minimax_rate_values():
    r = ghosal_white_noise_optimal_rate(1.0, 1000)
    assert abs(r["estimate"] - 1000 ** (-2.0 / 3.0)) < 1e-15
    assert abs(r["exponent"] - 2.0 / 3.0) < 1e-15
