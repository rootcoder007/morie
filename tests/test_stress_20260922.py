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
        src = f.read_text(encoding="utf-8")
        if not (re.search(r"^def cheatsheet\(|^cheatsheet\s*=", src, re.M)
                or re.search(r"^from \.[A-Za-z0-9_]+ import \(?[^)]*\bcheatsheet\b", src, re.M)):
            missing.append(f.stem)
    assert not missing, missing[:20]


def test_module_level_random_has_generator_surface():
    np.random.seed(3)
    out = np.random.negative_binomial(3, 0.4, size=10).tolist()
    assert len(out) == 10 and all(v >= 0 for v in out)
    assert len(np.random.standard_t(4, size=5).tolist()) == 5
    with pytest.raises(AttributeError):
        np.random.no_such_distribution


# ------------------------------------------------------------------ hunt round 2/3
def test_hunt_round_two_array():
    assert math.isnan(np.median(np.array([1.0, nan, 3.0])))
    assert np.median(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), axis=0).tolist() == [2.5, 3.5, 4.5]
    assert np.prod(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=1).tolist() == [2.0, 12.0]
    assert np.log2(np.array([0.0])).tolist() == [-math.inf]
    assert np.log10(np.array([0.0, -1.0])).tolist()[0] == -math.inf
    assert np.cosh(np.array([-1e300])).tolist() == [math.inf]
    r = np.rint(np.array([1.4, nan, math.inf])).tolist()
    assert r[0] == 1.0 and math.isnan(r[1]) and r[2] == math.inf
    assert np.exp2(np.array([1e300])).tolist() == [math.inf]
    u = np.unique(np.array([3.0, nan, 1.0, 3.0])).tolist()
    assert u[:2] == [1.0, 3.0] and math.isnan(u[2]) and len(u) == 3
    assert math.isnan(np.ptp(np.array([1.0, nan])))
    assert np.trunc(np.array([-1.7, 2.2])).tolist() == [-1.0, 2.0]
    assert np.transpose(np.array([[1.0, 2.0], [3.0, 4.0]])).tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert np.ravel(np.array([[1.0, 2.0], [3.0, 4.0]])).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert np.reciprocal(np.array([2.0, 0.0])).tolist() == [0.5, math.inf]
    assert np.cbrt(np.array([-8.0, 27.0])).tolist() == [-2.0, 3.0]
    assert np.arctanh(np.array([0.5, 1.0, 2.0])).tolist()[1] == math.inf
    assert math.isnan(np.arctanh(np.array([2.0])).tolist()[0])
    assert math.isnan(np.percentile(np.array([1.0, nan, 3.0]), 50))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert math.isnan(np.nanmean(np.array([])))
        assert math.isnan(np.median(np.array([])))


def test_hunt_round_two_frame():
    s = pd.Series([1.0, nan, 4.0, 2.0])
    assert s.cummax().tolist()[3] == 4.0 and math.isnan(s.cummax().tolist()[1])
    assert s.cumprod().tolist()[3] == 8.0
    assert s.ffill().tolist() == [1.0, 1.0, 4.0, 2.0]
    assert s.bfill().tolist() == [1.0, 4.0, 4.0, 2.0]
    assert pd.Series([1.0, nan, 3.0, nan]).interpolate().tolist() == [1.0, 2.0, 3.0, 3.0]
    assert s.first_valid_index() == 0 and s.argmax() == 2 and s.argmin() == 0
    assert s.nlargest(2).tolist() == [4.0, 2.0] and s.nsmallest(1).tolist() == [1.0]
    assert s.between(1.5, 4.0).tolist() == [False, False, True, True]
    assert s.where([True, True, False, True], 0.0).tolist()[2] == 0.0
    assert pd.Series([1.0, 2.0, 2.0]).is_monotonic_increasing
    assert not pd.Series([1.0, nan]).is_monotonic_increasing
    sk = pd.Series([1.0, 2.0, 3.0, 10.0]).skew()
    assert abs(sk - 1.763632614803888) < 1e-9  # pandas value
    ku = pd.Series([1.0, 2.0, 3.0, 10.0]).kurt()
    assert abs(ku - 3.228) < 1e-9  # pandas value


def test_hunt_round_two_stats():
    assert st.chi2.ppf(1.0, 3) == math.inf and st.t.ppf(0.0, 5) == -math.inf
    assert math.isnan(st.norm.ppf(1.5))
    assert st.geom.cdf(-1, 0.3) == 0.0 and st.geom.pmf(0, 0.3) == 0.0
    assert st.beta.logpdf(2.5, 2, 3) == -math.inf
    assert st.binom.pmf(2.5, 10, 0.3) == 0.0 and st.binom.cdf(math.inf, 10, 0.3) == 1.0
    assert st.nbinom.ppf(0.0, 3, 0.4) == -1.0
    assert abs(st.genextreme.ppf(1.0, 0.1) - 10.0) < 1e-12
    # exact noncentral t (scipy reference values)
    assert abs(st.nct.cdf(1.0, 5, 1.0) - 0.480926141) < 1e-8
    assert abs(st.nct.pdf(0.5, 5, 1.0) - 0.336004555) < 1e-7
    assert abs(st.nct.ppf(0.975, 5, 1.0) - 4.313082542) < 1e-6
    assert abs(st.nct.sf(10.0, 5, 1.0) - 0.000668678) < 1e-8


# ------------------------------------------------------------------ hunt round 4
def test_hunt_round_four_stats():
    x = [2.1, 3.4, 3.4, 5.6, 1.2, 4.4, 4.4, 6.0]
    y = [3.3, 2.2, 5.5, 4.1, 4.1, 7.0, 1.0, 2.0]
    r = st.linregress(x, y)
    slope, intercept, rv, pv, se = r  # the shape twelve fn modules unpack
    assert abs(slope - (-0.16703297)) < 1e-6 and abs(intercept - 4.28681319) < 1e-6
    assert abs(rv - (-0.13921217)) < 1e-6 and abs(pv - 0.74233) < 1e-4 and se > 0
    assert math.isnan(st.pearsonr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])[1])
    assert abs(st.wilcoxon(x, y)[1] - 0.875) < 1e-9  # exact, scipy
    assert abs(st.kendalltau(x, y)[1] - 0.52677416) < 1e-6  # tie-corrected, scipy
    assert abs(st.kendalltau([1, 2, 3, 4, 5], [1, 3, 2, 5, 4])[1] - 0.2333333) < 1e-6  # exact
    assert abs(st.ks_2samp(x, y)[1] - 0.98010878) < 1e-6
    assert st.mode([1, 2, 2, 3]).mode == 2 and st.mode([1, 2, 2, 3]).count == 2
    assert abs(st.power_divergence([10, 20, 30])[0] - 10.0) < 1e-12
    assert abs(st.power_divergence([10, 20, 30], lambda_="log-likelihood")[0] - 10.464962875) < 1e-6
    assert abs(st.combine_pvalues([0.01, 0.2, 0.5])[1] - 0.0316) < 2e-3
    assert abs(st.entropy([0.2, 0.3, 0.5]) - 1.0296530) < 1e-6
    assert abs(st.entropy([0.5, 0.5], base=2) - 1.0) < 1e-12
    m, sd = st.norm.fit(x)
    assert abs(m - sum(x) / 8) < 1e-12 and sd > 0
    assert st.expon.fit(x)[0] == 1.2
    assert abs(float(st.gaussian_kde(x)(3.0)[0]) - 0.17374396) < 1e-6


def test_hunt_round_four_frame():
    df = pd.DataFrame({"g": ["a", "b", "a", "b", "a"], "x": [1.0, 2.0, nan, 4.0, 5.0],
                       "y": [10.0, 20.0, 30.0, 40.0, 50.0]})
    assert df.sort_values("x", ascending=False)["y"].tolist() == [50.0, 40.0, 20.0, 10.0, 30.0]
    assert df.sort_values("x", na_position="first")["y"].tolist() == [30.0, 10.0, 20.0, 40.0, 50.0]
    vc = df["x"].value_counts(dropna=False)
    assert len(vc) == 5 and any(math.isnan(k) for k in vc.index)
    assert math.isnan(df["x"].sum(skipna=False))
    assert df[["x", "y"]].max(axis=1).tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert df.pivot_table(index="g", values="y", aggfunc="mean")["y"].tolist() == [30.0, 30.0]
    assert df.melt(id_vars="g", value_vars=["x", "y"])["value"].tolist()[5:] == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert list(df.set_index("g").reset_index().columns) == ["g", "x", "y"]
    assert df["g"].duplicated().tolist() == [False, False, True, True, True]
    r = df["y"].rolling(2).mean().tolist()
    assert math.isnan(r[0]) and r[1:] == [15.0, 25.0, 35.0, 45.0]
    assert df["y"].expanding().sum().tolist() == [10.0, 30.0, 60.0, 100.0, 150.0]
    assert len(df.sample(3, random_state=1)) == 3
    agg = df.groupby("g")["y"].agg(["mean", "max"])
    assert agg["max"].tolist() == [50.0, 40.0] and agg["mean"].tolist() == [30.0, 30.0]
    assert df.groupby("g")["x"].first().tolist() == [1.0, 2.0]
    assert pd.qcut(df["y"], 2, labels=False).tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_hunt_round_four_array_and_ml():
    assert math.isnan(np.power(np.array([-8.0]), 1.0 / 3).tolist()[0])
    with pytest.raises(ValueError):
        np.mean(np.array([1.0, 2.0]), axis=1)
    assert np.mean(np.array([1.0, 2.0]), axis=0) == 1.5
    from morie.fn._ml_core import MinMaxScaler

    out = MinMaxScaler().fit_transform(np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])).tolist()
    assert out == [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]


# --- round three (2026-09-23, at 4e87add61) ---------------------------------


def test_round_three_simod_estimates():
    from morie.fn.simod import simod

    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 2))
    b = np.array([3.0, -1.0])
    b = b / np.linalg.norm(b)
    y = np.sin(X @ b) + rng.normal(0, 0.2, 200)
    r = simod(y, X, max_iter=200)
    assert r["n_iter"] > 0
    assert abs(float(np.array(r["beta"]) @ b)) > 0.95


def test_round_three_scalar_in_scalar_out():
    from morie.fn.gelua import gelua
    from morie.fn.kmswig import swish
    from morie.fn.qnorm import qnorm

    assert isinstance(qnorm(0.5), float)
    assert qnorm(0.5) == pytest.approx(0.0, abs=1e-12)
    assert isinstance(gelua(0.5), float)
    assert isinstance(swish(0.5), float)
    assert len(qnorm([0.5, 0.975])) == 2


def test_round_three_correlate_shorter_first():
    a, v = np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0])
    assert np.correlate(a, v).tolist() == [11.0, 8.0, 5.0]
    assert np.correlate(v, a).tolist() == [5.0, 8.0, 11.0]
    assert np.correlate(np.array([1.0, 2.0, 3.0]),
                        np.array([0.0, 1.0, 0.5])).tolist() == [3.5]


def test_round_three_twoprop_degenerate_is_zero_and_one():
    import morie

    with pytest.warns(UserWarning, match="degenerate"):
        r = morie.mrm_twoprop_test(0, 50, 0, 50)
    assert (r.chi2, r.p_value_chi2, r.p_value_fisher) == (0.0, 1.0, 1.0)


def test_round_three_exit_codes_for_failed_backends(monkeypatch):
    from morie import llm, perseus, runner

    monkeypatch.setattr(perseus, "detect_available_provider", lambda: "ollama")
    monkeypatch.setattr(perseus, "llm_ask",
                        lambda *a, **k: llm._FallbackText("nobody home"))
    payload = perseus.ask_percy("hi", use_agent=False, stream=False)
    assert payload["mode"] == "local_fallback"
    assert runner._llm_exit_code(payload) == 1

    monkeypatch.setattr(perseus, "llm_ask",
                        lambda *a, **k: iter([llm._FallbackText("nobody home")]))
    payload = perseus.ask_percy("hi", use_agent=False, stream=True)
    assert payload["mode"] == "local_fallback"
    assert list(payload["output_stream"]) == ["nobody home"]

    failed = {"mode": "agent", "output_text": "FreeAPI request failed: x",
              "failed": True}
    assert runner._llm_exit_code(failed) == 1
    assert runner._llm_exit_code({"mode": "agent", "output_text": "ok",
                                  "failed": False}) == 0
