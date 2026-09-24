"""Tests for halft.half_life."""

import math

from morie.fn import _array_core as np

from morie.fn.halft import half_life


def test_halft_basic():
    """Test basic functionality."""
    result = half_life("CCO", Vd=50.0, Cl=5.0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] > 0


def test_halft_edge():
    """Test edge cases."""
    result = half_life(
        "CCO",
        Vd=70.0,
        Cl=7.0,
        route="two_compartment",
        V1=40.0,
        V2=30.0,
        Q=10.0,
    )
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] > 0
