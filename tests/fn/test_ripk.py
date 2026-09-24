"""Tests for ripK.ripley_k_function."""

import math

from morie.fn import _array_core as np
from morie.fn.ripk import ripley_k_function


def test_ripk_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    points = rng.uniform(0.0, 10.0, (n, 2))
    window = (0.0, 10.0, 0.0, 10.0)
    r = np.linspace(0.5, 5.0, 10)
    result = ripley_k_function(points, window, r)
    assert isinstance(result, dict)
    for key in ("r", "k", "k_border", "l", "csr",
                "lambda_hat", "area", "n", "method"):
        assert key in result
    assert len(result["r"]) == 10
    assert result["n"] == n
    assert math.isfinite(result["lambda_hat"])


def test_ripk_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 2
    points = rng.uniform(0.0, 5.0, (n, 2))
    window = (0.0, 5.0, 0.0, 5.0)
    r = 1.0
    result = ripley_k_function(points, window, r)
    assert isinstance(result, dict)
    assert result["n"] == n
    assert math.isfinite(result["lambda_hat"])
    assert "k" in result and "l" in result and "csr" in result
