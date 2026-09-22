"""Tests for ca11e19.ca_chapter_11_equation_19."""

from morie.fn import _array_core as np

from morie.fn.ca11e19 import ca_chapter_11_equation_19


def test_ca11e19_basic():
    """Test basic functionality with a single positive se_ln_or."""
    se_ln_or = 0.5
    result = ca_chapter_11_equation_19(se_ln_or)
    assert isinstance(result, dict)
    assert "value" in result
    # Independent computation from the documented formula:
    # se_d = sqrt(se_ln_or^2 / 1.65^2)
    expected = np.sqrt(se_ln_or ** 2 / 1.65 ** 2)
    assert abs(result["value"] - expected) < 1e-12
    # method is part of the documented payload
    assert result["method"] == "Weisburd et al. (2022) eq. (11.19)"


def test_ca11e19_edge():
    """Test edge cases."""
    # Smaller positive se_ln_or
    se_ln_or = 0.1
    result = ca_chapter_11_equation_19(se_ln_or)
    assert isinstance(result, dict)
    expected = np.sqrt(se_ln_or ** 2 / 1.65 ** 2)
    assert abs(result["value"] - expected) < 1e-12
    assert "value" in result

    # Larger positive se_ln_or
    se_ln_or = 2.0
    result = ca_chapter_11_equation_19(se_ln_or)
    assert isinstance(result, dict)
    expected = np.sqrt(se_ln_or ** 2 / 1.65 ** 2)
    assert abs(result["value"] - expected) < 1e-12
