"""Tests for otdwd.ot_doubly_stoch_proj."""

import math
from morie.fn import _array_core as np

from morie.fn.otdwd import ot_doubly_stoch_proj


def test_otdwd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    base = rng.normal(0, 1, (10, 10))
    K = np.abs(base) + 1.0
    result = ot_doubly_stoch_proj(K, max_iter=200)
    assert isinstance(result, dict)
    assert "M" in result
    assert "iters" in result
    assert "d1" in result
    assert "d2" in result
    assert result["n"] == 10
    assert result["iters"] == 200
    assert math.isfinite(result["row_err"])
    assert math.isfinite(result["col_err"])
    assert result["row_err"] < 1e-3
    assert result["col_err"] < 1e-3


def test_otdwd_edge():
    """Test edge cases."""
    K = [[1.0, 2.0], [3.0, 4.0]]
    result = ot_doubly_stoch_proj(K, max_iter=50)
    assert isinstance(result, dict)
    assert "M" in result
    assert result["n"] == 2
    assert math.isfinite(result["row_err"])
    assert math.isfinite(result["col_err"])
    assert result["row_err"] < 1e-3
    assert result["col_err"] < 1e-3
