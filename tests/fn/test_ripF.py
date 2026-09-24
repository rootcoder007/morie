"""Tests for ripF.ripley_f_function."""

import math

from morie.fn import _array_core as np

from morie.fn.ripF import ripley_f_function


def test_ripF_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    window = [0.0, 10.0, 0.0, 10.0]
    points = rng.uniform(0.0, 10.0, (n, 2))
    r = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    result = ripley_f_function(points, window, r)
    for key in ("r", "f", "f_border", "csr", "m", "lambda_hat", "n", "method"):
        assert key in result
    assert result["n"] == n
    assert result["m"] == 400
    assert len(result["f"]) == len(r)
    assert len(result["f_border"]) == len(r)
    assert len(result["csr"]) == len(r)
    for v in result["f"]:
        assert 0.0 <= v <= 1.0
    for v in result["f_border"]:
        assert 0.0 <= v <= 1.0
    for v in result["csr"]:
        assert 0.0 <= v <= 1.0
    assert math.isfinite(result["lambda_hat"])
    assert result["lambda_hat"] > 0.0


def test_ripF_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 5
    window = [0.0, 5.0, 0.0, 5.0]
    points = rng.uniform(0.0, 5.0, (n, 2))
    r = [0.5]
    result = ripley_f_function(points, window, r)
    for key in ("r", "f", "f_border", "csr", "m", "lambda_hat", "n", "method"):
        assert key in result
    assert result["n"] == n
    assert result["m"] == 400
    assert len(result["f"]) == 1
    assert len(result["f_border"]) == 1
    assert len(result["csr"]) == 1
    assert 0.0 <= result["f"][0] <= 1.0
    assert 0.0 <= result["f_border"][0] <= 1.0
    assert 0.0 <= result["csr"][0] <= 1.0
    assert math.isfinite(result["lambda_hat"])
