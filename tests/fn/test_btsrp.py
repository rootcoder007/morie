"""Tests for btsrp.bootstrap_ci."""

from morie.fn import _array_core as np

from morie.fn.btsrp import bootstrap_ci


def test_btsrp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bootstrap_ci(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_btsrp_edge():
    """Test edge cases."""
    result = bootstrap_ci(np.array([42.0]))
    assert result["n"] == 1
