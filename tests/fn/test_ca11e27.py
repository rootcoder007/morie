"""Tests for ca11e27.ca_chapter_11_equation_27."""

import math

from morie.fn import _array_core as np

from morie.fn.ca11e27 import ca_chapter_11_equation_27


def test_ca11e27_basic():
    """Test basic functionality with a scalar se_d value."""
    rng = np.random.default_rng(42)
    x = float(rng.normal(0, 1, 1)[0])
    result = ca_chapter_11_equation_27(x)
    assert isinstance(result, dict)
    assert "value" in result
    expected = math.sqrt(x ** 2 / 0.606 ** 2)
    assert result["value"] == expected
    assert result["method"] == "Weisburd et al. (2022) eq. (11.27)"


def test_ca11e27_edge():
    """Test edge case with a known scalar se_d value."""
    x = 0.606
    result = ca_chapter_11_equation_27(x)
    assert isinstance(result, dict)
    assert "value" in result
    expected = math.sqrt(x ** 2 / 0.606 ** 2)
    assert result["value"] == expected
