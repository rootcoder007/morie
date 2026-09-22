"""Tests for ca9e11.ca_chapter_9_equation_11."""

from morie.fn import _array_core as np

from morie.fn.ca9e11 import ca_chapter_9_equation_11


def test_ca9e11_basic():
    """Test basic functionality."""
    m1, m2 = 5.0, 4.5
    s1, s2 = 1.2, 1.3
    n1, n2 = 30, 35
    result = ca_chapter_9_equation_11(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
    assert "t" in result
    # Verify the formula directly from the docstring.
    df = (n1 + n2) - 2
    t_expected = (m1 - m2) / np.sqrt(
        ((s1**2 * (n1 - 1) + s2**2 * (n2 - 1)) / df)
        * ((n1 + n2) / (n1 * n2))
    )
    assert result["t"] == t_expected


def test_ca9e11_edge():
    """Test edge cases."""
    m1, m2 = 10.0, 10.0
    s1, s2 = 1.0, 1.0
    n1, n2 = 20, 25
    result = ca_chapter_9_equation_11(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
    assert "t" in result
    # Equal means should give t == 0.
    assert result["t"] == 0.0
