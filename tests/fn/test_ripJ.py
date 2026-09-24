"""Tests for ripJ.ripley_j_function."""

from morie.fn import _array_core as np
from morie.fn.ripJ import ripley_j_function


def test_ripJ_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    points = rng.normal(0, 1, (40, 2))
    window = (-3.0, -3.0, 3.0, 3.0)
    r = np.linspace(0.1, 3.0, 10)
    result = ripley_j_function(points, window, r)
    assert isinstance(result, dict)
    assert "r" in result
    assert "j" in result
    assert "g" in result
    assert "f" in result
    assert "j_csr" in result
    assert "lambda_est" in result
    assert "n_defined" in result
    assert "method" in result
    assert len(result["r"]) == 10
    assert len(result["j"]) == 10
    assert len(result["j_csr"]) == 10


def test_ripJ_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    points = rng.normal(0, 1, (40, 2))
    result = ripley_j_function(points)
    assert isinstance(result, dict)
    assert "r" in result
    assert "j" in result
    assert "n_defined" in result
    assert isinstance(result["n_defined"], int)
    assert len(result["r"]) == len(result["j"])
