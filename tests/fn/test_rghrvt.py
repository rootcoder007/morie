"""Tests for rghrvt.rangayyan_hrv_time_domain."""

from morie.fn import _array_core as np
from morie.fn.bsaqrs import rangayyan_hrv_time_domain


def test_rghrvt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    rr = rng.uniform(0.5, 1.5, 40)
    result = rangayyan_hrv_time_domain(rr)
    assert isinstance(result, dict)
    assert "n" in result
    assert result["n"] == len(rr)


def test_rghrvt_edge():
    """Test edge cases."""
    rr = np.array([1.0, 2.0])
    result = rangayyan_hrv_time_domain(rr)
    assert isinstance(result, dict)
    assert "n" in result
    assert result["n"] == 2
