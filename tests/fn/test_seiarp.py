"""Tests for seiarp.seira_asymptomatic."""

import math

from morie.fn import _array_core as np

from morie.fn.seiarp import seira_asymptomatic


def test_seiarp_basic():
    """Test basic functionality."""
    S = 990.0
    E = 5.0
    I = 3.0
    A = 1.0
    R = 1.0
    params = [0.5, 0.2, 0.1, 0.8, 0.5, 0.1]
    result = seira_asymptomatic(S, E, I, A, R, params)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0.0


def test_seiarp_edge():
    """Test edge cases."""
    S = 999.0
    E = 0.0
    I = 1.0
    A = 0.0
    R = 0.0
    params = [0.5, 0.2, 0.1, 1.0, 0.5, 0.1]
    result = seira_asymptomatic(S, E, I, A, R, params)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0.0
