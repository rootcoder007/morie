"""Tests for dpqua.dp_quantile."""

from morie.fn import _array_core as np
from morie.fn.dpqua import dp_quantile


def test_dpqua_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_quantile(x)
    assert isinstance(result, dict)
    assert "release" in result


def test_dpqua_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_quantile(x)
    assert isinstance(result, dict)


def test_exponential_mechanism_probabilities_recomputed():
    """P(gap i) proportional to exp(eps u(i)/2) * gap width, u(i) = -|i - q n|."""
    import math

    import pytest

    x = [2.0, 5.0, 3.0, 9.0, 4.0]
    a, b, eps, q = 0.0, 10.0, 1.2, 0.4
    edges = [a] + sorted(x) + [b]
    n = 5
    logp = [eps * (-abs(i - q * n)) / 2 + math.log(max(edges[i + 1] - edges[i], 1e-300)) for i in range(n + 1)]
    m = max(logp)
    p = [math.exp(v - m) for v in logp]
    s = sum(p)
    r = dp_quantile(x, q=q, epsilon=eps, a=a, b=b, seed=3)
    assert [float(v) for v in r["probabilities"]] == pytest.approx([v / s for v in p], rel=1e-12)
    lo, hi = r["interval"]
    assert lo <= r["release"] <= hi
