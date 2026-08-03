"""Tests for Ghosal Ch 13 survival modules."""
import math

from morie.fn.gh_c13_1 import ghosal_surv_dp_post
from morie.fn.gh_c13_3 import ghosal_beta_proc_def
from morie.fn.gh_c13_4 import ghosal_bp_discrete
from morie.fn.gh_c13_5 import ghosal_bp_cont
from morie.fn.gh_c13_6 import ghosal_bp_path_gen
from morie.fn.gh_c13_7 import ghosal_mix_bp
from morie.fn.gh_c13_8 import ghosal_ntr_def
from morie.fn.gh_c13_9 import ghosal_ntr_levy
from morie.fn.gh_c13_10 import ghosal_ntr_consist
from morie.fn.gh_c13_11 import ghosal_ntr_bvm
from morie.fn.gh_c13_12 import ghosal_smhaz_gp
from morie.fn.gh_c13_13 import ghosal_cox_model
from morie.fn.gh_c13_14 import ghosal_cox_post
from morie.fn.gh_c13_16 import ghosal_bb_censored

TIMES = [0.2, 0.5, 0.7, 1.1, 1.5]
EVENTS = [1, 0, 1, 1, 0]


def test_censored_dp_survival_monotone():
    s1 = ghosal_surv_dp_post(TIMES, EVENTS, 0.6)["estimate"]
    s2 = ghosal_surv_dp_post(TIMES, EVENTS, 1.2)["estimate"]
    assert 0 < s2 <= s1 <= 1


def test_beta_process_nondecreasing():
    r = ghosal_beta_proc_def([0.2, 0.4, 0.6, 0.8, 1.0])
    assert r["nondecreasing"] is True


def test_bp_discrete_prior_mean():
    r = ghosal_bp_discrete()
    assert r["prior_mean_gap"] < 0.02


def test_bp_levy_mass_identity():
    r = ghosal_bp_cont()
    assert r["gap"] < 1e-3


def test_bp_jump_path():
    r = ghosal_bp_path_gen()
    assert r["pure_jump_nondecreasing"] is True


def test_mix_bp_mean():
    r = ghosal_mix_bp()
    assert abs(r["estimate"] - (0.5 + 1.0 + 2.0) / 3.0) < 1e-12


def test_ntr_cdf_valid():
    r = ghosal_ntr_def([0.1, 0.3, 0.2, 0.5])
    assert r["nondecreasing"] is True
    expected = 1.0 - math.exp(-1.1)
    assert abs(r["estimate"] - expected) < 1e-12


def test_ntr_laplace_exact():
    # single atom: exp(-m (1 - e^{-f}))
    r = ghosal_ntr_levy([1.0], [2.0])
    assert abs(r["estimate"]
               - math.exp(-2.0 * (1.0 - math.exp(-1.0)))) < 1e-12


def test_ntr_consistency_improves():
    r = ghosal_ntr_consist()
    assert r["improving"] is True


def test_ntr_bvm_variance():
    r = ghosal_ntr_bvm()
    assert r["gap"] < 0.05


def test_gp_hazard_recovers_constant():
    r = ghosal_smhaz_gp()
    assert r["estimate"] < 0.5


def test_cox_proportionality_exact():
    r = ghosal_cox_model()
    assert r["proportional"] is True


def test_cox_posterior_recovers_beta():
    r = ghosal_cox_post()
    assert r["error"] < 0.3


def test_bb_censored_equals_km():
    # events at 0.2 (risk 5) and 0.7 (risk 3): S = 4/5 * 2/3
    r = ghosal_bb_censored(TIMES, EVENTS, 1.0)
    assert abs(r["estimate"] - 4.0 / 5.0 * 2.0 / 3.0) < 1e-12
