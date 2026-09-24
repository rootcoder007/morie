"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner3e39.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_39."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner3e39 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_39,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e39_basic():
    """Test basic functionality."""
    var_x = 9.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_39(var_x)
    # The function returns a RichResult (or dict-like) with 'variance' and 'sd' keys.
    assert isinstance(result, dict)
    assert "variance" in result
    assert "sd" in result
    assert result["variance"] == var_x
    sd = result["sd"]
    assert isinstance(sd, float)
    assert math.isfinite(sd)
    assert sd >= 0.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e39_edge():
    """Test edge cases."""
    var_x = 0.0
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_39(var_x)
    assert isinstance(result, dict)
    assert "sd" in result
    sd = result["sd"]
    assert math.isfinite(sd)
    assert sd == 0.0
