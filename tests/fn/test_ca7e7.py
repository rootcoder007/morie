"""Tests for ca7e7.ca_chapter_7_equation_7."""

from morie.fn import _array_core as np

from morie.fn.ca7e7 import ca_chapter_7_equation_7


def test_ca7e7_basic():
    """Test basic functionality."""
    sigma2_u = 1.5
    sigma2_e = 2.5
    result = ca_chapter_7_equation_7(sigma2_u, sigma2_e)
    assert isinstance(result, dict)
    assert "value" in result
    expected = sigma2_u / (sigma2_u + sigma2_e)
    assert result["value"] == expected
    assert result["method"] == "Weisburd et al. (2022) eq. (7.7)"


def test_ca7e7_edge():
    """Test edge cases."""
    sigma2_u = 0.0
    sigma2_e = 1.0
    result = ca_chapter_7_equation_7(sigma2_u, sigma2_e)
    assert isinstance(result, dict)
    assert "value" in result
    expected = sigma2_u / (sigma2_u + sigma2_e)
    assert result["value"] == expected
