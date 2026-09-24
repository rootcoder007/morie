"""Tests for otmxh.ot_mixture_w2."""

import math
from morie.fn import _array_core as np

from morie.fn.otmxh import ot_mixture_w2


def test_otmxh_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    K, d = 3, 2
    mus1 = rng.normal(0, 1, (K, d))
    mus2 = rng.normal(0, 1, (K, d))
    Sigmas1 = [np.eye(d) for _ in range(K)]
    Sigmas2 = [np.eye(d) for _ in range(K)]
    w1 = rng.uniform(0, 1, K)
    w2 = rng.uniform(0, 1, K)
    result = ot_mixture_w2(mus1, Sigmas1, w1, mus2, Sigmas2, w2)
    assert isinstance(result, dict)
    assert "MW2" in result
    assert "MW2_sq" in result
    assert "T" in result
    assert "C" in result
    assert "K1" in result
    assert "K2" in result
    assert "d" in result
    assert math.isfinite(result["MW2"])
    assert result["MW2"] >= 0
    assert result["MW2_sq"] >= 0
    assert result["K1"] == K
    assert result["K2"] == K
    assert result["d"] == d


def test_otmxh_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    K, d = 2, 1
    mus1 = rng.normal(0, 1, (K, d))
    mus2 = rng.normal(0, 1, (K, d))
    Sigmas1 = [[[1.0]] for _ in range(K)]
    Sigmas2 = [[[1.0]] for _ in range(K)]
    w1 = rng.uniform(0, 1, K)
    w2 = rng.uniform(0, 1, K)
    result = ot_mixture_w2(mus1, Sigmas1, w1, mus2, Sigmas2, w2)
    assert isinstance(result, dict)
    assert "MW2" in result
    assert math.isfinite(result["MW2"])
    assert result["MW2"] >= 0
    assert result["K1"] == K
    assert result["K2"] == K
    assert result["d"] == d
