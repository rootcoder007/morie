"""Tests for ca7e5.ca_chapter_7_equation_5."""

from morie.fn import _array_core as np

from morie.fn.ca7e5 import ca_chapter_7_equation_5


def test_ca7e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    beta00 = rng.normal(0, 1)
    u_j = rng.normal(0, 1)
    result = ca_chapter_7_equation_5(beta00, u_j)
    assert isinstance(result, dict)
    assert "value" in result
    expected = float(beta00) + float(u_j)
    assert result["value"] == expected


def test_ca7e5_edge():
    """Test edge cases."""
    beta00 = 0.0
    u_j = 0.0
    result = ca_chapter_7_equation_5(beta00, u_j)
    assert isinstance(result, dict)
    assert result["value"] == 0.0
    assert result["value"] == float(beta00) + float(u_j)
