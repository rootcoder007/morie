"""Tests for fzc2x.fauzi_c2_coefficient."""

from morie.fn import _array_core as np

from morie.fn.fzc2x import fauzi_c2_coefficient


def test_fzc2x_basic():
    """Test basic functionality."""
    # For the identity transformation g, dg=1, d2g=0, d3g=0, so the formula
    # collapses to f''(x) -- the classical KDE bias coefficient.
    dg = 1.0
    d2g = 0.0
    d3g = 0.0
    density = 0.4
    fp = 0.1
    fpp = -0.3
    result = fauzi_c2_coefficient(dg, d2g, d3g, density, fp, fpp)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "scaled" in result
    assert "method" in result

    expected = d3g * density + 3.0 * d2g * dg * fp + dg ** 3 * fpp
    assert result["estimate"] == expected
    assert result["scaled"] == expected / dg


def test_fzc2x_edge():
    """Test edge cases."""
    dg = 1.0
    d2g = 0.0
    d3g = 0.0
    density = 0.4
    fp = 0.1
    fpp = -0.3
    result = fauzi_c2_coefficient(dg, d2g, d3g, density, fp, fpp)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "scaled" in result
