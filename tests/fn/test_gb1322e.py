"""Tests for gb1322e.gibbons_efficacy."""

from morie.fn import _array_core as np

from morie.fn.gb1322e import gibbons_efficacy


def test_gb1322e_basic():
    """Test basic functionality against the documented formula."""
    deriv = 4.0
    var = 8.0
    result = gibbons_efficacy(deriv, var)

    expected = deriv * deriv / var
    assert isinstance(result, dict)
    assert "efficacy" in result
    assert "deriv" in result
    assert "var" in result
    assert "method" in result
    assert result["efficacy"] == expected
    assert result["deriv"] == float(deriv)
    assert result["var"] == float(var)


def test_gb1322e_edge():
    """Test that var must be strictly positive."""
    deriv = 1.0
    var = 1.0
    result = gibbons_efficacy(deriv, var)

    expected = deriv * deriv / var
    assert isinstance(result, dict)
    assert result["efficacy"] == expected
    assert result["deriv"] == float(deriv)
    assert result["var"] == float(var)
    assert result["method"].startswith("efficacy e(T)")
