"""Tests for causovlap.causal_overlap_diagnostic."""

from morie.fn import _array_core as np

from morie.fn.causovlap import causal_overlap_diagnostic


def test_causovlap_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    ps = rng.uniform(0, 1, 200)
    treat = [1.0 if rng.uniform(0, 1) < p else 0.0 for p in ps]
    result = causal_overlap_diagnostic(ps, treat)
    assert isinstance(result, dict)
    assert "common_support" in result
    assert "n_outside" in result
    assert "prop_extreme" in result
    assert "min_treated_ps" in result
    assert "max_control_ps" in result
    assert "overlap_coefficient" in result
    overlap = result["overlap_coefficient"]
    assert 0.0 <= overlap <= 1.0
    lo, hi = result["common_support"]
    assert lo <= hi


def test_causovlap_edge():
    """Test edge cases with custom bins and eps."""
    rng = np.random.default_rng(42)
    ps = rng.uniform(0.1, 0.9, 200)
    treat = [1.0 if rng.uniform(0, 1) < p else 0.0 for p in ps]
    result = causal_overlap_diagnostic(ps, treat, bins=10, eps=0.1)
    assert isinstance(result, dict)
    assert "common_support" in result
    assert "n_outside" in result
    assert "prop_extreme" in result
    assert "min_treated_ps" in result
    assert "max_control_ps" in result
    assert "overlap_coefficient" in result
    lo, hi = result["common_support"]
    assert lo <= hi
    n_outside = result["n_outside"]
    assert 0 <= n_outside <= len(ps)
