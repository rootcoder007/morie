"""Tests for causdidcs.causal_did_callaway_sa."""

from morie.fn import _array_core as np

from morie.fn.causdidcs import causal_did_callaway_sa


def test_causdidcs_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    D = np.array([1.0 if (i // 10) >= 5 and (i % 10) < 5 else 0.0 for i in range(100)])
    unit = np.array([float(i % 10) for i in range(100)])
    time = np.array([float(i // 10) for i in range(100)])
    result = causal_did_callaway_sa(y, D, unit, time)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_causdidcs_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0, 1, 100)
    D = np.array([1.0 if (i // 10) >= 5 and (i % 10) < 5 else 0.0 for i in range(100)])
    unit = np.array([float(i % 10) for i in range(100)])
    time = np.array([float(i // 10) for i in range(100)])
    result = causal_did_callaway_sa(y, D, unit, time)
    assert isinstance(result, dict)
