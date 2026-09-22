"""Tests for ca11e17.ca_chapter_11_equation_17."""

import math

from morie.fn import _array_core as np

from morie.fn.ca11e17 import ca_chapter_11_equation_17


def test_ca11e17_basic():
    """Test basic functionality with a single positive se_lnOR value."""
    se_ln_or = 0.5
    result = ca_chapter_11_equation_17(se_ln_or)
    assert isinstance(result, dict)
    assert "value" in result
    # Formula: se_d = sqrt(se_lnOR^2 / (pi^2/3))
    expected = math.sqrt(se_ln_or ** 2 / (math.pi ** 2 / 3))
    assert math.isclose(result["value"], expected)
    assert result.get("method") == "Weisburd et al. (2022) eq. (11.17)"


def test_ca11e17_edge():
    """Test edge case with a larger se_lnOR value."""
    se_ln_or = 1.2
    result = ca_chapter_11_equation_17(se_ln_or)
    assert isinstance(result, dict)
    assert "value" in result
    expected = math.sqrt(se_ln_or ** 2 / (math.pi ** 2 / 3))
    assert math.isclose(result["value"], expected)
