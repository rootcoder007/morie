"""Tests for icc2k.icc_two_way_random_avg."""

from morie.fn import _array_core as np

from morie.fn.icc2k import icc_two_way_random_avg


def test_icc2k_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.array([float(i // 4) for i in range(100)])
    rater = np.array([float(i % 4) for i in range(100)])
    result = icc_two_way_random_avg(y, subject, rater)
    assert isinstance(result, dict)
    assert "value" in result
def test_icc2k_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    subject = np.array([float(i // 4) for i in range(100)])
    rater = np.array([float(i % 4) for i in range(100)])
    result = icc_two_way_random_avg(y, subject, rater)
    assert isinstance(result, dict)
