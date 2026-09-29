"""Tests for mannK.mann_kendall."""

from morie.fn import _array_core as np
from morie.fn.mannK import mann_kendall


def test_mannK_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mann_kendall(x)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_mannK_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mann_kendall(x)
    assert isinstance(result, dict)


def test_mannK_recomputed_with_ties():
    import math

    x = [1.2, 0.8, 1.9, 2.4, 2.1, 3.3, 3.0, 4.1, 2.4, 3.3, 3.3]
    n = len(x)
    s = sum((x[j] > x[i]) - (x[j] < x[i]) for i in range(n) for j in range(i + 1, n))
    v = (n * (n - 1) * (2 * n + 5) - (2 * 1 * 9) - (3 * 2 * 11)) / 18
    r = mann_kendall(x)
    assert r["S"] == s and abs(r["varS"] - v) < 1e-12
    assert abs(r["statistic"] - (abs(s) - 1) / math.sqrt(v)) < 1e-12
    assert abs(mann_kendall(x, continuity=False)["statistic"] - s / math.sqrt(v)) < 1e-12
    assert abs(r["p_value"] - math.erfc(r["statistic"] / math.sqrt(2))) < 1e-12
