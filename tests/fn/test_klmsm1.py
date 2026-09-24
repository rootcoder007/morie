"""Tests for klmsm1.kl_molecular_smooth."""

import math

from morie.fn import _array_core as np

from morie.fn.klmsm1 import kl_molecular_smooth


def test_klmsm1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    V = 50
    p = list(rng.integers(1, 20, V))
    q = list(rng.integers(1, 20, V))
    eps = 0.1
    result = kl_molecular_smooth(p, q, eps)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "kl_pq" in result
    assert "kl_qp" in result
    assert "symmetric_kl" in result
    assert result["vocabulary"] == float(V)
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0


def test_klmsm1_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    V = 2
    p = list(rng.integers(0, 5, V))
    q = list(rng.integers(0, 5, V))
    eps = 1e-3
    result = kl_molecular_smooth(p, q, eps)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0
