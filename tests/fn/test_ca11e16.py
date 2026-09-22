"""Tests for ca11e16.ca_chapter_11_equation_16."""

from morie.fn import _array_core as np

from morie.fn.ca11e16 import ca_chapter_11_equation_16


def test_ca11e16_basic():
    """Test basic functionality with a scalar ln(OR)."""
    ln_or = 0.5
    result = ca_chapter_11_equation_16(ln_or)
    assert isinstance(result, dict)
    assert "value" in result
    # d = ln(OR) / sqrt(pi^2/3)
    expected = ln_or / np.sqrt(np.pi ** 2 / 3)
    assert np.isclose(result["value"], expected)
    # Method key per the documented reference
    assert result["method"] == "Weisburd et al. (2022) eq. (11.16)"


def test_ca11e16_zero():
    """Test that ln(OR)=0 yields d=0."""
    result = ca_chapter_11_equation_16(0.0)
    assert isinstance(result, dict)
    assert np.isclose(result["value"], 0.0)


def test_ca11e16_known_value():
    """Test against an independently-computed expected value."""
    ln_or = 1.0
    result = ca_chapter_11_equation_16(ln_or)
    expected = 1.0 / np.sqrt(np.pi ** 2 / 3)
    assert np.isclose(result["value"], expected)
    assert np.isclose(result["value"], ln_or / 1.814, rtol=1e-3)


def test_ca11e16_negative():
    """Test with a negative ln(OR)."""
    ln_or = -0.7
    result = ca_chapter_11_equation_16(ln_or)
    expected = ln_or / np.sqrt(np.pi ** 2 / 3)
    assert result["value"] < 0
    assert np.isclose(result["value"], expected)
