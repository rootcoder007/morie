"""Tests for otsinkit.ot_sinkhorn_iter_count."""

from morie.fn import _array_core as np

from morie.fn.otsinkit import ot_sinkhorn_iter_count


def test_otsinkit_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    a = rng.uniform(0.1, 1.0, n)
    s = np.sum(a)
    a = [float(x) / s for x in a]
    b = rng.uniform(0.1, 1.0, n)
    s = np.sum(b)
    b = [float(x) / s for x in b]
    C = rng.uniform(0.0, 1.0, (n, n))
    epsilon = 0.1
    tol = 1e-3
    max_iter = 100
    result = ot_sinkhorn_iter_count(a, b, C, epsilon, tol, max_iter)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "reached" in result
    assert "final_error" in result
    assert "trace" in result
    assert "method" in result
    import math
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= float(max_iter)
    assert isinstance(result["reached"], bool)
    assert math.isfinite(result["final_error"])
    assert len(result["trace"]) == max_iter


def test_otsinkit_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 30
    a = rng.uniform(0.1, 1.0, n)
    s = np.sum(a)
    a = [float(x) / s for x in a]
    b = rng.uniform(0.1, 1.0, n)
    s = np.sum(b)
    b = [float(x) / s for x in b]
    C = rng.uniform(0.0, 1.0, (n, n))
    epsilon = 0.5
    tol = 1e-2
    # Use default max_iter (omitted) and a larger epsilon to help convergence.
    result = ot_sinkhorn_iter_count(a, b, C, epsilon, tol)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "reached" in result
    assert "trace" in result
    assert len(result["trace"]) == 200
