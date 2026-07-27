"""Tests for causbalt.causal_balance_test (Austin 2009)."""

import numpy as np
import pytest

from morie.fn.causbalt import causal_balance_test


def _sample(seed=0, n=400, shift=0.0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, 3)) + shift * t[:, None]
    return X, t


def test_balanced_data_is_balanced():
    X, t = _sample(seed=1, shift=0.0)
    res = causal_balance_test(X, t)
    assert res["max_smd"] < 0.2
    assert res["n_imbalanced"] <= 1


def test_a_known_shift_is_recovered_as_the_smd():
    """A shift of c on unit-variance data gives SMD ~ c."""
    X, t = _sample(seed=2, n=4000, shift=0.8)
    res = causal_balance_test(X, t)
    assert res["smd"][0] == pytest.approx(0.8, abs=0.1)
    assert not res["balanced"]


def test_equal_weights_reproduce_the_unweighted_result():
    """The weighted variance reduces to the n-1 form at equal weights."""
    X, t = _sample(seed=3, shift=0.5)
    a = causal_balance_test(X, t)["smd"]
    b = causal_balance_test(X, t, weights=np.full(X.shape[0], 2.5))["smd"]
    assert np.allclose(a, b, rtol=1e-12)


def test_weighting_can_restore_balance():
    """Weights that undo a confounded assignment shrink the SMD."""
    rng = np.random.default_rng(4)
    n = 4000
    x = rng.normal(0, 1, n)
    ps = 1 / (1 + np.exp(-x))          # treatment depends on x
    t = (rng.random(n) < ps).astype(int)
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))   # inverse probability
    X = x.reshape(-1, 1)
    before = causal_balance_test(X, t)["max_smd"]
    after = causal_balance_test(X, t, weights=w)["max_smd"]
    assert before > 0.3
    assert after < before / 2


def test_smd_sign_follows_the_direction_of_the_difference():
    X, t = _sample(seed=5, n=2000, shift=-0.6)
    assert causal_balance_test(X, t)["smd"][0] < 0


def test_threshold_controls_the_verdict():
    X, t = _sample(seed=6, n=2000, shift=0.3)
    assert not causal_balance_test(X, t, threshold=0.1)["balanced"]
    assert causal_balance_test(X, t, threshold=0.9)["balanced"]


def test_no_p_value_unless_a_cdf_is_supplied():
    X, t = _sample(seed=7)
    assert causal_balance_test(X, t)["p_value"] is None
    from scipy import stats

    assert causal_balance_test(X, t, cdf=stats.norm().cdf)["p_value"] is not None


def test_validates_inputs():
    X, t = _sample(seed=8)
    with pytest.raises(ValueError, match="treat must be binary"):
        causal_balance_test(X, np.arange(X.shape[0]))
    with pytest.raises(ValueError, match="one entry per row"):
        causal_balance_test(X, t[:-1])
    with pytest.raises(ValueError, match="weights must be finite"):
        causal_balance_test(X, t, weights=-np.ones(X.shape[0]))
    with pytest.raises(ValueError, match="X must be finite"):
        bad = X.copy(); bad[0, 0] = np.nan
        causal_balance_test(bad, t)
