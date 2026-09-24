"""The art of doing mathematics consists in finding that special case which contains all the germs of generality. — David Hilbert"""

import math
from morie.fn import _array_core as np
from morie.fn.hubsa import hits_hubs_authorities


def test_hubsa_basic():
    """Test basic functionality."""
    n = 20
    rng_y = np.random.default_rng(43)
    y = [abs(v) + 0.1 for v in rng_y.normal(0, 1, n)]
    rng_A = np.random.default_rng(42)
    A = [[abs(v) + 0.1 for v in row] for row in rng_A.normal(0, 1, (n, n))]
    tol = 1e-6
    result = hits_hubs_authorities(y, A, tol)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "hubs" in result
    assert "authorities" in result
    assert "iterations" in result
    assert "converged" in result
    assert "n" in result
    assert result["n"] == n
    assert len(result["hubs"]) == n
    assert len(result["authorities"]) == n
    assert math.isfinite(result["estimate"])
    assert isinstance(result["converged"], bool)
    assert isinstance(result["iterations"], int)
    assert result["iterations"] >= 0


def test_hubsa_edge():
    """Test edge cases."""
    n = 3
    rng_y = np.random.default_rng(43)
    y = [abs(v) + 0.1 for v in rng_y.normal(0, 1, n)]
    rng_A = np.random.default_rng(42)
    A = [[abs(v) + 0.1 for v in row] for row in rng_A.normal(0, 1, (n, n))]
    result = hits_hubs_authorities(y, A, 1e-12)
    assert isinstance(result, dict)
    assert "hubs" in result
    assert "authorities" in result
    assert result["n"] == n
    assert len(result["hubs"]) == n
    assert len(result["authorities"]) == n
    assert math.isfinite(result["estimate"])
    assert isinstance(result["converged"], bool)
    assert isinstance(result["iterations"], int)
