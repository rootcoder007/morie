# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the 2026-09-22 stress-test findings.

Each test is the reviewer's reproducer turned into an assertion; the
numbers quoted in comments are what the previous code returned.
"""
from __future__ import annotations

import math
import warnings

import pytest

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn import _stats_core as st

nan = float("nan")


# ------------------------------------------------------------------ 3
def test_logistic_refuses_nan_and_has_no_cutoff():
    from morie.fn._ml_core import LogisticRegression, _sigmoid

    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    Xn = [r[:] for r in X]
    Xn[1][0] = nan
    with pytest.raises(ValueError, match="NaN"):
        LogisticRegression().fit(np.array(Xn), np.array(y))
    assert _sigmoid(-40.0) > 0.0 and _sigmoid(-40.0) < 1e-15  # was exactly 0.0
    assert math.isnan(_sigmoid(nan))
    assert _sigmoid(800.0) == 1.0 and _sigmoid(-800.0) == 0.0


def test_propensity_scores_name_missing_columns():
    import morie

    df = pd.DataFrame({"x": [0.1, nan, 0.3, 0.4], "z": [1.0, 2.0, 3.0, 4.0],
                       "t": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="'x'"):
        morie.compute_propensity_scores(df, "t", ["x", "z"])


# ------------------------------------------------------------------ 4
def test_gini_is_nan_on_missing_values():
    import morie

    vals = [float(v) for v in range(1, 31)]
    clean = float(morie.fairness_gini(vals))
    assert 0.3 < clean < 0.4
    v2 = vals[:]
    v2[7] = nan
    r = morie.fairness_gini(v2)
    assert math.isnan(float(r))  # was 0.0, "relatively evenly spread"
    assert "undefined" in str(r)
    assert float(morie.fairness_gini([10.0] * 30)) == 0.0


# ------------------------------------------------------------------ 5
@pytest.mark.parametrize("v", [[1.0, nan, 3.0], [nan, 1.0, 3.0], [3.0, 1.0, nan]])
def test_max_min_propagate_nan_regardless_of_position(v):
    assert math.isnan(np.max(np.array(v)))
    assert math.isnan(np.min(np.array(v)))
    assert np.argmax(np.array(v)) == v.index(next(x for x in v if x != x))


def test_sort_puts_nan_last_and_stays_sorted():
    assert np.sort(np.array([3.0, nan, 1.0, 2.0])).tolist()[:3] == [1.0, 2.0, 3.0]
    assert math.isnan(np.sort(np.array([3.0, nan, 1.0, 2.0])).tolist()[3])
    assert np.argsort(np.array([3.0, nan, 1.0])).tolist() == [2.0, 0.0, 1.0]
    assert np.sort(np.array([[3.0, 1.0], [2.0, 0.0]])).tolist() == [[1.0, 3.0], [0.0, 2.0]]


def test_sign_angle_trace_nanargmax():
    s = np.sign(np.array([1.0, nan, -1.0])).tolist()
    assert s[0] == 1.0 and math.isnan(s[1]) and s[2] == -1.0
    assert math.isnan(np.angle(np.array([nan])).tolist()[0])
    with pytest.raises(ValueError):
        np.trace(np.array([5.0, 6.0, 7.0]))
    with pytest.raises(ValueError):
        np.nanargmax(np.array([]))
    with pytest.raises(ValueError):
        np.nanargmax(np.array([nan, nan]))
    assert np.nanargmax(np.array([nan, 2.0, 1.0])) == 1


# ------------------------------------------------------------------ 6
def test_nanstd_nanvar_honour_axis():
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.nanstd(m, axis=0).tolist() == [1.5, 1.5, 1.5]
    assert np.nanvar(m, axis=0).tolist() == [2.25, 2.25, 2.25]
    assert np.nanvar(m, axis=1, keepdims=True).tolist() == [[2.0 / 3.0], [2.0 / 3.0]]
    assert abs(float(np.nanstd(m)) - 1.707825127659933) < 1e-12


# ------------------------------------------------------------------ 7
def test_spacing_runs_and_is_signed():
    assert np.spacing(1.0) == 2.220446049250313e-16
    assert np.spacing(-1.0) == -2.220446049250313e-16
    assert math.isnan(np.spacing(nan))
    assert np.spacing(np.array([1.0, 2.0])).tolist() == [2.220446049250313e-16, 4.440892098500626e-16]


# ------------------------------------------------------------------ 8
def test_oneprop_test_runs_and_matches_docs():
    import morie

    r = morie.mrm_oneprop_test(40, 100, 0.5)
    assert abs(r.p_value_exact - 0.05688793) < 1e-6  # docs/designexptr_coverage.md
    assert 0.30 < r.ci95_exact_lower < 0.31 and 0.50 < r.ci95_exact_upper < 0.51
    ci = st.binomtest(40, 100).proportion_ci(method="wilson")
    assert 0.30 < ci.low < 0.31 and 0.49 < ci.high < 0.51


# ------------------------------------------------------------------ 9
@pytest.mark.parametrize("d", ["uniform", "norm", "expon", "poisson", "gamma", "beta", "t", "chi2",
                               "geom", "nbinom", "hypergeom", "binom", "laplace", "logistic"])
def test_clt_demo_and_rvs_for_every_distribution(d):
    import morie

    kw = {"gamma": {"a": 2.0}, "beta": {"a": 2.0, "b": 3.0}, "t": {"df": 5}, "chi2": {"df": 3},
          "geom": {"p": 0.3}, "nbinom": {"n": 3, "p": 0.4}, "hypergeom": {"M": 20, "n": 7, "N": 12},
          "binom": {"n": 10, "p": 0.3}, "poisson": {"mu": 2.0}}.get(d, {})
    out = morie.mrm_clt_demo(d, n_samples=20, sample_size=8, **kw)
    assert len(out) == 20
    g = np.random.default_rng(3)
    assert np.random.default_rng(g) is g
    x = getattr(st, d).rvs(size=50, random_state=g, **kw)
    assert len(x.tolist()) == 50 and all(v == v for v in x.tolist())


def test_rvs_inverse_transform_is_distributed_right():
    g = np.random.default_rng(11)
    x = st.expon.rvs(size=4000, random_state=g).tolist()
    assert abs(sum(x) / len(x) - 1.0) < 0.06
    k = st.geom.rvs(size=4000, random_state=g, p=0.25).tolist()
    assert min(k) >= 1 and abs(sum(k) / len(k) - 4.0) < 0.3


# ------------------------------------------------------------------ 10
def test_series_rank_keeps_nan_as_nan():
    r = pd.Series([10.0, nan, 30.0, 20.0]).rank().tolist()
    assert r[0] == 1.0 and math.isnan(r[1]) and r[2] == 3.0 and r[3] == 2.0
    assert all(math.isnan(v) for v in pd.Series([nan, nan]).rank().tolist())
    assert pd.Series([10.0, nan, 30.0]).rank(na_option="top").tolist() == [2.0, 1.0, 3.0]
    assert pd.Series([10.0, nan, 30.0]).rank(na_option="bottom").tolist() == [1.0, 3.0, 2.0]
    assert pd.Series([1.0, 1.0, 2.0]).rank(method="min").tolist() == [1.0, 1.0, 3.0]
    assert pd.Series([1.0, 1.0, 2.0]).rank(method="dense").tolist() == [1.0, 1.0, 2.0]
    assert pd.Series([1.0, 1.0, 2.0]).rank(method="first").tolist() == [1.0, 2.0, 3.0]


# ------------------------------------------------------------------ 11
def test_tox_missing_is_not_censored():
    from morie.tox import tox_left_censor_impute

    r = tox_left_censor_impute([0.4, nan, 0.9, 0.02], lod=0.05)
    assert r["censored"].tolist() == [False, False, False, True]
    assert r["n_missing"] == 1
    assert abs(r["fraction_censored"] - 1 / 3) < 1e-12
    assert math.isnan(r["imputed"].tolist()[1])


# ------------------------------------------------------------------ 14/15 array shim
def test_generator_surface_and_axis_keywords():
    g = np.random.default_rng(5)
    for name, args in [("standard_t", (4,)), ("triangular", (0.0, 0.5, 1.0)), ("weibull", (1.5,)),
                       ("pareto", (3.0,)), ("power", (2.0,)), ("rayleigh", (1.0,)), ("gumbel", ()),
                       ("logistic", ()), ("wald", (1.0, 1.0)), ("vonmises", (0.0, 2.0)),
                       ("f", (3, 7)), ("noncentral_chisquare", (3, 1.5)), ("negative_binomial", (3, 0.4)),
                       ("hypergeometric", (7, 13, 5)), ("zipf", (2.5,)), ("logseries", (0.3,)),
                       ("standard_cauchy", ()), ("standard_exponential", ())]:
        out = getattr(g, name)(*args, size=30).tolist()
        assert len(out) == 30 and all(v == v for v in out), name
    m = g.multinomial(20, [0.2, 0.3, 0.5]).tolist()
    assert sum(m) == 20 and len(m) == 3
    assert np.cumsum(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0).tolist() == [[1.0, 2.0], [4.0, 6.0]]
    assert np.cumsum(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=1).tolist() == [[1.0, 3.0], [3.0, 7.0]]


def test_shapes_follow_numpy():
    rr, cc = np.nonzero(np.array([[0.0, 1.0], [2.0, 0.0]]))
    assert rr.tolist() == [0.0, 1.0] and cc.tolist() == [1.0, 0.0]
    assert np.tile(np.array([[1.0, 2.0], [3.0, 4.0]]), 2).tolist() == [[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]]
    assert np.kron(np.array([1.0, 2.0]), np.array([1.0, 10.0])).tolist() == [1.0, 10.0, 2.0, 20.0]
    assert np.ediff1d(np.array([[1.0, 4.0], [9.0, 16.0]])).tolist() == [3.0, 5.0, 7.0]
    assert np.squeeze(np.array([7.0])) == 7.0
    assert np.flip(np.array([[1.0, 2.0], [3.0, 4.0]])).tolist() == [[4.0, 3.0], [2.0, 1.0]]
    assert np.argsort(np.array([[3.0, 1.0], [0.0, 2.0]])).tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_domain_errors_are_nan_or_inf():
    assert math.isnan(np.arcsin(2.0))
    assert math.isnan(np.log10(-1.0))
    assert math.isnan(np.floor(nan))
    assert np.ceil(math.inf) == math.inf
    assert np.divide(1.0, 0.0) == math.inf and np.divide(-1.0, 0.0) == -math.inf
    assert math.isnan(np.divide(0.0, 0.0))
    assert math.isnan(np.mod(np.array([1.0]), 0.0).tolist()[0])
    assert math.isnan(np.sum(np.array([math.inf, -math.inf])))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert math.isnan(np.mean(np.array([])))
    assert any(issubclass(x.category, RuntimeWarning) for x in w)


def test_frame_corr_and_idx():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
    c = df.corr()
    assert c["a"].tolist()[0] == 1.0 and math.isnan(c["b"].tolist()[1])
    with pytest.raises(ValueError):
        pd.Series([nan, nan]).idxmax()
    with pytest.raises(ValueError):
        pd.Series([nan, nan]).idxmin()


def test_parse_version_prerelease_sorts_below_release():
    from morie._update_check import _parse_version as pv

    assert pv("2.0.0rc1") < pv("2.0.0") < pv("2.0.1")
    assert pv("1.3.2") > pv("1.3.1") and pv("1.3.2.post1") > pv("1.3.2")


def test_every_fn_module_has_cheatsheet():
    import re
    from pathlib import Path

    import morie.fn as fn

    missing = []
    for f in sorted(Path(fn.__path__[0]).glob("*.py")):
        if f.name.startswith("_") or f.name == "describe.py":
            continue
        if not re.search(r"^def cheatsheet\(|^cheatsheet\s*=", f.read_text(), re.M):
            missing.append(f.stem)
    assert not missing, missing[:20]
