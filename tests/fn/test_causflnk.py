"""Tests for causflnk.causal_falsification_test."""

from morie.fn import _array_core as np

from morie.fn.causflnk import causal_falsification_test


def test_causflnk_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    y_pre = rng.normal(0, 1, 100)
    treat = rng.integers(0, 2, 100).astype(float)
    X_baseline = rng.normal(0, 1, 100)
    result = causal_falsification_test(y_pre, treat, X_baseline)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "p_value" in result
    assert "passed" in result
    assert "min_detectable_effect" in result
    assert "power_note" in result
    # Independent computation of the OLS coefficient on `treat` (no covariates),
    # checking that the adjusted estimate shifts when baseline covariates are added.
    treat_only = causal_falsification_test(y_pre, treat)
    Xb = np.atleast_2d(np.asarray(X_baseline, dtype=float))
    A = np.column_stack([np.ones(y_pre.size), treat, Xb.T])
    beta, *_ = np.linalg.lstsq(A, y_pre, rcond=None)
    assert abs(result["estimate"] - float(beta[1])) < 1e-8
    assert result["estimate"] != treat_only["estimate"]


def test_causflnk_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    y_pre = rng.normal(0, 1, 100)
    treat = rng.integers(0, 2, 100).astype(float)
    X_baseline = rng.normal(0, 1, 100)
    result = causal_falsification_test(y_pre, treat, X_baseline)
    assert isinstance(result, dict)
    assert "power_note" in result
    # Minimum detectable effect is documented as 2.8 * se.
    assert abs(result["min_detectable_effect"] - 2.8 * result["se"]) < 1e-12
