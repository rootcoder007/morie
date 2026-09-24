"""Tests for hampw.hampel_three_part."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.hampw import hampel_three_part


def test_hampw_basic():
    """Test basic functionality with normal-distributed residuals."""
    rng = np.random.default_rng(43)
    y = list(rng.normal(0, 1, 100))
    a, b, c = 2.0, 4.0, 8.0
    result = hampel_three_part(y, a=a, b=b, c=c)
    assert isinstance(result, dict)
    for key in ("estimate", "weights", "n_zero", "n", "a", "b", "c", "method"):
        assert key in result
    assert result["n"] == 100
    assert len(result["weights"]) == 100
    assert all(0.0 <= w <= 1.0 for w in result["weights"])
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert result["a"] == a
    assert result["b"] == b
    assert result["c"] == c
    expected_n_zero = sum(1 for e in y if abs(e) > c)
    assert result["n_zero"] == expected_n_zero


def test_hampw_edge():
    """Test that an empty residual vector raises ValueError."""
    with pytest.raises(ValueError):
        hampel_three_part([])
