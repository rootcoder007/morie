"""Tests for hampel.hampel_redescend."""

import math

from morie.fn import _array_core as np

from morie.fn.hampel import hampel_redescend


def test_hampel_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    r = rng.normal(0, 1, 50)
    result = hampel_redescend(r, a=2.0, b=4.0, c=8.0)
    assert isinstance(result, dict)
    for key in ("estimate", "psi", "psi_deriv", "n_reject",
                "n", "a", "b", "c", "method"):
        assert key in result
    assert result["n"] == 50
    assert len(result["psi"]) == 50
    assert len(result["psi_deriv"]) == 50
    assert result["n_reject"] >= 0
    assert result["a"] == 2.0
    assert result["b"] == 4.0
    assert result["c"] == 8.0
    assert math.isfinite(result["estimate"])


def test_hampel_edge():
    """Test edge cases."""
    r = [0.0, 1.0, 3.0, 5.0, 10.0]
    result = hampel_redescend(r, a=2.0, b=4.0, c=8.0)
    assert isinstance(result, dict)
    assert result["n"] == 5
    assert result["n_reject"] == 1
    assert len(result["psi"]) == 5
    assert len(result["psi_deriv"]) == 5
    # psi: |r|<=a -> r; a<|r|<=b -> a*sign; b<|r|<=c -> a*(c-|r|)/(c-b)*sign; else 0
    #   0  -> 0
    #   1  -> 1
    #   3  -> 2
    #   5  -> 2*(8-5)/(8-4) = 1.5
    #   10 -> 0 (rejected)
    assert result["estimate"] == (0.0 + 1.0 + 2.0 + 1.5 + 0.0) / 5
    assert math.isfinite(result["estimate"])
