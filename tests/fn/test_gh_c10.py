"""Tests for Ghosal Ch 10 adaptation modules."""
import math

from morie.fn.gh_c10_1 import ghosal_adapt_thm
from morie.fn.gh_c10_2 import ghosal_univ_weights
from morie.fn.gh_c10_3 import ghosal_param_rate
from morie.fn.gh_c10_4 import ghosal_two_model_adp
from morie.fn.gh_c10_5 import ghosal_wn_adapt
from morie.fn.gh_besov_prior import ghosal_besov_prior
from morie.fn.gh_c10_6 import ghosal_rnd_series_pr
from morie.fn.gh_c10_8 import ghosal_frs_reg
from morie.fn.gh_c10_9 import ghosal_frs_binreg
from morie.fn.gh_c10_10 import ghosal_frs_poireg
from morie.fn.gh_c10_11 import ghosal_func_reg
from morie.fn.gh_c10_12 import ghosal_modsel_bic
from morie.fn.gh_c10_14 import ghosal_param_np_bf


def test_adaptive_prior_finds_dimension():
    r = ghosal_adapt_thm()
    assert r["estimate"] == 3.0


def test_universal_weights_regimes():
    ok = ghosal_univ_weights(c=2.0, eps_scale=1.0)
    assert ok["converges"] is True
    bad = ghosal_univ_weights(c=0.5, eps_scale=1.0)
    assert bad["converges"] is False


def test_parametric_adaptation_rate():
    r = ghosal_param_rate()
    assert r["parametric"] is True


def test_two_model_small_wins_under_small_truth():
    r = ghosal_two_model_adp()
    assert r["small_model_wins"] is True
    assert r["estimate"] > 0.9


def test_spike_slab_inclusion():
    r = ghosal_wn_adapt()
    inc = r["inclusion_probs"]
    assert inc[0] > 0.95 and inc[1] > 0.95      # true signals
    assert max(inc[2:]) < 0.5                    # nulls


def test_besov_prior_finite():
    r = ghosal_besov_prior()
    assert r["finite"] is True
    assert r["n_active"] > 0


def test_random_series_K_concentrates():
    r = ghosal_rnd_series_pr()
    assert abs(r["mode_K"] - 4) <= 1


def test_frs_regression_risk_small():
    r = ghosal_frs_reg()
    assert r["estimate"] < 0.05
    assert 1 <= r["K_hat"] <= 8


def test_frs_binary_orders():
    r = ghosal_frs_binreg()
    assert r["orders_correctly"] is True


def test_frs_poisson_recovers():
    r = ghosal_frs_poireg()
    assert r["estimate"] < 0.35


def test_functional_regression_recovers():
    r = ghosal_func_reg()
    assert r["estimate"] < 0.15


def test_bayes_factor_both_directions():
    h1 = ghosal_modsel_bic(True)
    assert h1["supports_H1"] is True
    h0 = ghosal_modsel_bic(False)
    assert h0["supports_H1"] is False


def test_param_np_bf_directions():
    par = ghosal_param_np_bf(parametric_truth=True)
    assert par["nonparametric_wins"] is False
    npt = ghosal_param_np_bf(parametric_truth=False)
    assert npt["nonparametric_wins"] is True
