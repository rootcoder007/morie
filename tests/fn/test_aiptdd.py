"""Tests for aiptdd.aipw_did (Sant'Anna & Zhao 2020)."""

import numpy as np
import pytest

from morie.fn.aiptdd import aipw_did


def _panel(seed=0, n=3000, att=2.0, linear_ps=True, linear_out=True):
    """Selection on X, and X also shifts the trend -- so a naive DiD is
    biased and the covariates have to do real work."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    eta = 0.8 * x if linear_ps else 1.2 * np.sin(2 * x)
    d = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    unit = rng.normal(0, 2, n)
    trend = 0.9 * x if linear_out else 1.1 * np.cos(2 * x)
    pre = unit + rng.normal(0, 1, n)
    post = unit + trend + att * d + rng.normal(0, 1, n)
    return pre, post, d, x.reshape(-1, 1)


def test_recovers_the_att():
    pre, post, d, x = _panel(seed=1, att=2.0)
    res = aipw_did(pre, post, d, x)
    assert res["att"] == pytest.approx(2.0, abs=0.2)


def test_naive_did_is_biased_here_but_aipw_is_not():
    """Confirms the fixture actually needs adjustment."""
    pre, post, d, x = _panel(seed=2, att=2.0)
    dy = post - pre
    naive = dy[d == 1].mean() - dy[d == 0].mean()
    assert abs(naive - 2.0) > 0.3, "fixture should confound the naive estimate"
    assert aipw_did(pre, post, d, x)["att"] == pytest.approx(2.0, abs=0.2)


def test_double_robustness_propensity_wrong_outcome_right():
    """The defining property: only one working model need be correct."""
    pre, post, d, x = _panel(seed=3, att=2.0, linear_ps=False, linear_out=True)
    assert aipw_did(pre, post, d, x)["att"] == pytest.approx(2.0, abs=0.25)


def test_double_robustness_outcome_wrong_propensity_right():
    pre, post, d, x = _panel(seed=4, att=2.0, linear_ps=True, linear_out=False)
    assert aipw_did(pre, post, d, x)["att"] == pytest.approx(2.0, abs=0.25)


def test_zero_effect_is_not_detected():
    pre, post, d, x = _panel(seed=5, att=0.0)
    assert aipw_did(pre, post, d, x)["p_value"] > 0.05


def test_a_real_effect_is_detected():
    pre, post, d, x = _panel(seed=6, att=2.0)
    assert aipw_did(pre, post, d, x)["p_value"] < 0.001


def test_confidence_interval_covers_the_truth():
    pre, post, d, x = _panel(seed=7, att=2.0)
    r = aipw_did(pre, post, d, x)
    assert r["ci_low"] < 2.0 < r["ci_high"]


def test_trimming_is_reported_not_hidden():
    """Extreme propensity scores are the practical failure mode."""
    rng = np.random.default_rng(8)
    n = 800
    x = rng.normal(0, 1, n)
    d = (x > -0.5).astype(float)      # near-deterministic assignment
    pre = rng.normal(0, 1, n)
    post = pre + 1.0 * d + rng.normal(0, 1, n)
    r = aipw_did(pre, post, d, x.reshape(-1, 1), trim=0.9)
    assert r["n_trimmed"] > 0
    assert r["ps_max"] <= 0.9 + 1e-12


def test_validates_inputs():
    pre, post, d, x = _panel(seed=9, n=200)
    with pytest.raises(ValueError, match="share a length"):
        aipw_did(pre, post[:-1], d, x)
    with pytest.raises(ValueError, match="must be binary"):
        aipw_did(pre, post, np.full_like(d, 2.0), x)
    with pytest.raises(ValueError, match="at least 2 units per arm"):
        aipw_did(pre, post, np.zeros_like(d), x)
    with pytest.raises(ValueError, match="must be finite"):
        bad = pre.copy(); bad[0] = np.nan
        aipw_did(bad, post, d, x)
    with pytest.raises(ValueError, match=r"trim must lie in \(0, 1\)"):
        aipw_did(pre, post, d, x, trim=1.0)
