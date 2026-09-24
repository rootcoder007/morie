"""Tests for km085.kamath_ch6_cbs_variance."""

import math
import pytest

from morie.fn import _array_core as np
from morie.fn.km085 import kamath_ch6_cbs_variance


def test_km085_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 5
    p = 3
    W = [f"w{i}" for i in range(n)]
    A = [f"a{j}" for j in range(p)]
    p_a = rng.uniform(1e-6, 1.0, size=(n, p))
    p_prior = rng.uniform(1e-6, 1.0, size=(n, p))
    result = kamath_ch6_cbs_variance(W, A, p_a, p_prior)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert isinstance(result["estimate"], float)
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0.0
    assert "per_word" in result
    assert isinstance(result["per_word"], list)
    assert len(result["per_word"]) == n
    assert "log_ratios" in result
    assert isinstance(result["log_ratios"], list)
    assert len(result["log_ratios"]) == n
    for row in result["log_ratios"]:
        assert isinstance(row, list)
        assert len(row) == p
    assert result["n"] == n
    assert result["ddof"] == 0
    assert "method" in result
    assert isinstance(result["method"], str)


def test_km085_edge():
    """Test edge cases."""
    # Empty list of template words must raise ValueError
    W = []
    A = ["a1", "a2"]
    p_a = np.zeros((0, 2))
    p_prior = np.zeros((0, 2))
    with pytest.raises(ValueError):
        kamath_ch6_cbs_variance(W, A, p_a, p_prior)
