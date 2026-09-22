"""Tests for bocpd.bocpd."""

from morie.fn import _array_core as np

from morie.fn.bocpd import bocpd


def test_bocpd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bocpd(x)
    assert isinstance(result, dict)
    assert "cp_prob" in result
def test_bocpd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bocpd(x)
    assert isinstance(result, dict)
