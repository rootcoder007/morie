"""Tests for hrzsms.horowitz_smoothed_max_score."""

from morie.fn import _array_core as np

from morie.fn.hrzsms import hrz_smoothed_max_score


def test_hrzsms_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    p = 3
    x = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    bandwidth = 0.3
    result = hrz_smoothed_max_score(x, y, bandwidth)
    assert isinstance(result, dict)


def test_hrzsms_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    n = 40
    p = 3
    x = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    bandwidth = 0.5
    result = hrz_smoothed_max_score(x, y, bandwidth)
    assert isinstance(result, dict)
