"""Tests for mdvtr.median_voter."""

from morie.fn import _array_core as np
from morie.fn.mdvtr import median_voter


def test_mdvtr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = median_voter(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_mdvtr_edge():
    """Test edge cases."""
    result = median_voter(np.array([42.0]))
    assert result["n"] == 1


def test_order_statistic_interval_and_density_se_recomputed():
    import math

    import pytest

    x = [2.0, 3.5, 1.0, 4.0, 5.5, 3.0, 2.5, 6.0, 4.5, 3.8, 2.2]
    n = 11
    xs = sorted(x)

    def cdf(k):
        return sum(math.comb(n, i) for i in range(k + 1)) / 2**n

    k = max(c for c in range(1, n // 2 + 1) if cdf(c - 1) <= 0.025)
    r = median_voter(x)
    assert (r["ci_exact_lower"], r["ci_exact_upper"]) == (xs[k - 1], xs[n - k])
    assert r["exact_coverage"] == pytest.approx(1 - 2 * cdf(k - 1), rel=1e-12)
    m = sum(x) / n
    s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    assert r["se_normal"] == pytest.approx(1.2533141373155003 * s / math.sqrt(n), rel=1e-12)
    assert r["estimate"] == xs[5]
