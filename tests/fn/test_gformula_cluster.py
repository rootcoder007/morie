"""G-formula cluster: causmrop, gforml, gctvc, sntmod, gmccsm, pluginM.

Linear-Gaussian DGPs (models correctly specified) with time-varying
confounding where the naive regression is provably biased; recovery
asserted as rates over seeds."""

from morie.fn import _array_core as np
import pytest

from morie.fn.causmrop import causal_robins_g_formula
from morie.fn.gctvc import g_computation_time_varying
from morie.fn.gforml import robins_g_formula
from morie.fn.gmccsm import g_methods_consistency
from morie.fn.pluginM import plug_in_mediation
from morie.fn.sntmod import sequential_target_models


def _point_dgp(seed, n=2500):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=n)
    A = (rng.random(n) < 1 / (1 + np.exp(-1.5 * L))).astype(float)
    y = 2.0 * A + 1.5 * L + rng.normal(scale=0.5, size=n)
    return y, A, L


def _tv_dgp(seed, n=3000):
    # L2 is affected by A1 and confounds A2; Y picks up L2, so the
    # always-vs-never effect is 1 + 1 + 1.0*0.7 = 2.7.
    rng = np.random.default_rng(seed)
    L1 = rng.normal(size=n)
    A1 = (rng.random(n) < 1 / (1 + np.exp(-L1))).astype(float)
    L2 = 0.5 * L1 + 0.7 * A1 + rng.normal(scale=0.7, size=n)
    A2 = (rng.random(n) < 1 / (1 + np.exp(-1.5 * L2))).astype(float)
    y = 1.0 * A1 + 1.0 * A2 + 1.0 * L2 + rng.normal(scale=0.5, size=n)
    return y, np.c_[A1, A2], np.c_[L1, L2]


def test_causmrop_point_ate():
    hits = 0
    for seed in range(8):
        y, A, L = _point_dgp(seed)
        out = causal_robins_g_formula(y, A, L)
        naive = y[A == 1].mean() - y[A == 0].mean()
        assert abs(naive - 2.0) > 0.5  # measured ~3.7
        hits += abs(out["ate"] - 2.0) < 0.15
    assert hits >= 7  # measured 8/8
    assert out["EY1"] - out["EY0"] == pytest.approx(out["ate"])


def test_gforml_time_varying_recovery():
    hits = 0
    for seed in range(6):
        y, A, L = _tv_dgp(seed)
        hi = robins_g_formula(y, A, L, [1, 1], n_mc=4000, seed=seed)
        lo = robins_g_formula(y, A, L, [0, 0], n_mc=4000, seed=seed)
        hits += abs((hi["estimate"] - lo["estimate"]) - 2.7) < 0.25
    assert hits >= 5  # measured 6/6


def test_gctvc_front_end_matches_gforml():
    y, A, L = _tv_dgp(0)
    out = g_computation_time_varying(y, A, L, n_mc=3000, seed=1)
    hi = robins_g_formula(y, A, L, [1, 1], n_mc=3000, seed=1)
    lo = robins_g_formula(y, A, L, [0, 0], n_mc=3000, seed=1)
    assert out["estimate"] == pytest.approx(hi["estimate"] - lo["estimate"])
    assert out["EY_always"] == pytest.approx(hi["estimate"])


def test_sntmod_ice_recovery():
    hits = 0
    for seed in range(6):
        y, A, L = _tv_dgp(seed)
        hi = sequential_target_models(y, A, L, intervention=1)
        lo = sequential_target_models(y, A, L, intervention=0)
        hits += abs((hi["estimate"] - lo["estimate"]) - 2.7) < 0.25
    assert hits >= 5  # measured 6/6
    with pytest.raises(ValueError):
        sequential_target_models([1.0, 2.0], [[0.5, 1]], [[0.0, 0.0]])


def test_sntmod_point_matches_gformula():
    y, A, L = _point_dgp(3)
    ice = sequential_target_models(y, A, L, 1)["estimate"] - sequential_target_models(y, A, L, 0)["estimate"]
    g = causal_robins_g_formula(y, A, L)
    # both are standardised linear fits; measured gap < 0.03
    assert ice == pytest.approx(g["ate"], abs=0.1)


def test_gmccsm_agreement_under_correct_spec():
    hits = 0
    for seed in range(8):
        y, A, L = _point_dgp(seed)
        out = g_methods_consistency(y, A, L, tau=0.3)
        hits += out["consistent"]
        for k in ("ate_gformula", "ate_ipw", "ate_aipw"):
            assert abs(out[k] - 2.0) < 0.5
    assert hits >= 7  # measured max_divergence ~0.03-0.1
    with pytest.raises(ValueError):
        g_methods_consistency([1.0], [1], [0.0], tau=-1)


def test_pluginM_product_method():
    # x -> m (0.8), m -> y (1.5), direct 0.7, confounder c on both.
    hits = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 2000
        c = rng.normal(size=n)
        x = 0.5 * c + rng.normal(size=n)
        m = 0.8 * x + 0.6 * c + rng.normal(scale=0.7, size=n)
        y = 0.7 * x + 1.5 * m + 0.5 * c + rng.normal(scale=0.7, size=n)
        out = plug_in_mediation(x, m, y, c=c)
        ok = abs(out["nie"] - 1.2) < 0.15 and abs(out["nde"] - 0.7) < 0.15
        hits += ok
        assert out["te"] == pytest.approx(out["nde"] + out["nie"])
    assert hits >= 7  # measured 8/8
    with pytest.raises(ValueError):
        plug_in_mediation([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])  # too few obs
