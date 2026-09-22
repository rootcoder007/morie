"""Tests for ca7e15.ca_chapter_7_equation_15."""

from morie.fn import _array_core as np

from morie.fn.ca7e15 import ca_chapter_7_equation_15


def test_ca7e15_basic():
    """Test basic functionality."""
    beta0 = 2.5
    u_0j = 0.75
    result = ca_chapter_7_equation_15(beta0, u_0j)
    assert isinstance(result, dict)
    assert "value" in result
    # Numerically reconstruct the documented formula beta_0j = beta0 + u_0j
    expected = float(beta0) + float(u_0j)
    assert result["value"] == expected


def test_ca7e15_edge():
    """Test edge cases."""
    beta0 = -1.0
    u_0j = 3.0
    result = ca_chapter_7_equation_15(beta0, u_0j)
    assert isinstance(result, dict)
    assert "value" in result
    expected = float(beta0) + float(u_0j)
    assert result["value"] == expected
