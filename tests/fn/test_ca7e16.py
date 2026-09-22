"""Tests for ca7e16.ca_chapter_7_equation_16."""

from morie.fn import _array_core as np

from morie.fn.ca7e16 import ca_chapter_7_equation_16


def test_ca7e16_basic():
    """Test basic functionality with scalar inputs."""
    rng = np.random.default_rng(42)
    beta1 = 1.5
    u_1j = 0.25
    result = ca_chapter_7_equation_16(beta1, u_1j)
    expected = float(beta1) + float(u_1j)
    assert isinstance(result, dict)
    assert "value" in result
    assert isinstance(result["value"], float)
    # Independent expression matching the documented formula
    assert result["value"] == expected
    # Method identifies the literature source
    assert "method" in result


def test_ca7e16_edge():
    """Test edge case with zero random effect."""
    rng = np.random.default_rng(42)
    beta1 = 0.0
    u_1j = 0.0
    result = ca_chapter_7_equation_16(beta1, u_1j)
    expected = 0.0 + 0.0
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == expected


def test_ca7e16_negative_random_effect():
    """Test with a negative random-effect term."""
    rng = np.random.default_rng(42)
    beta1 = 2.0
    u_1j = -0.75
    result = ca_chapter_7_equation_16(beta1, u_1j)
    expected = float(beta1) + float(u_1j)
    assert result["value"] == expected
