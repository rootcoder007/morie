"""Tests for Ghosal Ch 14 partition/feature-process modules."""
import math

from morie.fn.gh_c14_1 import ghosal_eppf_def
from morie.fn.gh_c14_2 import ghosal_ewens_esf
from morie.fn.gh_c14_3 import ghosal_crp_def
from morie.fn.gh_c14_4 import ghosal_crf_def
from morie.fn.gh_c14_5 import ghosal_ssp_def
from morie.fn.gh_c14_6 import ghosal_ssp_post
from morie.fn.gh_c14_7 import ghosal_ssp_mix
from morie.fn.gh_c14_8 import ghosal_gibbs_proc
from morie.fn.gh_c14_9 import ghosal_py_process
from morie.fn.gh_py_univ_seq import ghosal_py_universal_sequence
from morie.fn.gh_c14_10 import ghosal_py_eppf
from morie.fn.gh_c14_11 import ghosal_py_powerlaw
from morie.fn.gh_c14_12 import ghosal_pk_process
from morie.fn.gh_c14_13 import ghosal_pk_levy
from morie.fn.gh_c14_14 import ghosal_nig_proc
from morie.fn.gh_c14_15 import ghosal_ncrm_def
from morie.fn.gh_c14_16 import ghosal_ncrm_levy
from morie.fn.gh_c14_17 import ghosal_disc_rp_rel
from morie.fn.gh_c14_18 import ghosal_ksbp_def
from morie.fn.gh_c14_19 import ghosal_local_dp
from morie.fn.gh_c14_20 import ghosal_probit_sbp
from morie.fn.gh_c14_21 import ghosal_ord_dep_sbp
from morie.fn.gh_c14_22 import ghosal_nested_dp
from morie.fn.gh_c14_23 import ghosal_ibp_def
from morie.fn.gh_c14_24 import ghosal_ibp_stickbr
from morie.fn.gh_c14_25 import ghosal_ibp_poisson


def test_eppf_symmetric_and_hand_value():
    r = ghosal_eppf_def([2, 1], alpha=1.0)
    # M=1, n=3: 1^2 * G(1) * G(2)G(1) / G(4) = 1/6
    assert abs(r["estimate"] - 1.0 / 6.0) < 1e-12
    assert r["symmetric"] is True


def test_ewens_hand_value():
    # n=2, one pair (m_2=1), alpha=1: 2!/(1*2) * 1/2/1 = 1/2
    r = ghosal_ewens_esf([0, 1], alpha=1.0)
    assert abs(r["estimate"] - 0.5) < 1e-12
    # all singletons n=2: 2!/2 * 1/1/2! = 1/2 (complements to 1)
    r2 = ghosal_ewens_esf([2], alpha=1.0)
    assert abs(r2["estimate"] - 0.5) < 1e-12


def test_crp_totals():
    r = ghosal_crp_def()
    assert r["total_seated"] == 100
    assert abs(r["estimate"] - r["expected_K_n"]) < 5.0


def test_crf_shares_dishes():
    assert ghosal_crf_def()["dishes_shared"] is True


def test_ssp_valid():
    r = ghosal_ssp_def([1, 2, 1], [0.0, 1.0, 2.0])
    assert abs(r["total_mass"] - 1.0) < 1e-12
    assert abs(r["estimate"] - 1.0) < 1e-12


def test_ssp_posterior_normalizes():
    r = ghosal_ssp_post([3, 2], alpha=1.0)
    assert abs(r["total"] - 1.0) < 1e-12
    assert abs(r["estimate"] - 1.0 / 6.0) < 1e-12


def test_ssp_mixture_positive():
    r = ghosal_ssp_mix([0.5], [1, 1], [0.3, 0.7])
    assert r["estimate"] > 0


def test_gibbs_type_dp_special_case():
    # d=0, V = alpha^k Gamma(alpha)/Gamma(alpha+n) reduces to Ewens
    r = ghosal_gibbs_proc([2, 1], V_n_k=1.0 / 6.0, discount=0.0)
    assert abs(r["estimate"] - 1.0 / 6.0 * 1.0) < 1e-12


def test_py_sticks_sum_to_one():
    r = ghosal_py_process()
    assert r["total_mass"] > 0.98


def test_py_stick_means_decrease():
    assert ghosal_py_universal_sequence()["decreasing"] is True


def test_py_eppf_reduces_to_dp_at_d0():
    py = ghosal_py_eppf([2, 1], d=0.0, theta=1.0)
    assert abs(py["estimate"] - 1.0 / 6.0) < 1e-12


def test_py_powerlaw_scale():
    r = ghosal_py_powerlaw()
    assert 0.5 < r["ratio"] < 2.0


def test_pk_normalizes():
    r = ghosal_pk_process()
    assert abs(r["total_mass"] - 1.0) < 1e-12


def test_pk_gamma_levy_mass():
    assert ghosal_pk_levy()["gap_to_one"] < 1e-3


def test_nig_levy_mass():
    r = ghosal_nig_proc()
    assert r["gap"] < 0.02


def test_ncrm_set_mass():
    r = ghosal_ncrm_def([1.0, 2.0, 3.0], [0.1, 0.4, 0.8])
    assert abs(r["estimate"] - 0.5) < 1e-12


def test_ncrm_laplace_exact():
    r = ghosal_ncrm_levy([1.0], [2.0], [1.0])
    assert abs(r["estimate"]
               - math.exp(-2.0 * (1.0 - math.exp(-1.0)))) < 1e-12


def test_hierarchy_py0_is_dp():
    assert ghosal_disc_rp_rel(0.0, 1.5)["py_reduces_to_dp"] is True


def test_ksbp_varies_with_x():
    r = ghosal_ksbp_def()
    assert r["weights_vary_with_x"] is True


def test_local_dp_restricts():
    r = ghosal_local_dp()
    assert r["local"] is True
    assert r["n_active"] > 0


def test_probit_sbp_mass():
    r = ghosal_probit_sbp()
    assert 0.9 < r["total_mass"] <= 1.0 + 1e-12


def test_ordering_dependent_nearest_wins():
    assert ghosal_ord_dep_sbp()["nearest_dominates"] is True


def test_nested_dp_clusters_groups():
    r = ghosal_nested_dp()
    assert 1 <= r["estimate"] <= 6


def test_ibp_dish_count_scale():
    r = ghosal_ibp_def()
    assert 0.3 * r["expected_dishes"] < r["estimate"] \
        < 3.0 * r["expected_dishes"]


def test_ibp_sticks_decrease():
    assert ghosal_ibp_stickbr()["decreasing"] is True


def test_ibp_poisson_mean():
    r = ghosal_ibp_poisson()
    assert r["gap"] < 1.5
