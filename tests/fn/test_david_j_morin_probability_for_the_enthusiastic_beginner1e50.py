"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e50.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_50."""

import math

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e50 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_50,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e50_basic():
    """Test basic functionality."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_50(5, 3)
    assert isinstance(result, dict)
    for key in ("n_picks", "n_types", "count", "count_alt", "forms_agree"):
        assert key in result
    assert result["n_picks"] == 5
    assert result["n_types"] == 3
    assert math.isfinite(result["count"])
    assert math.isfinite(result["count_alt"])
    assert int(result["count"]) == int(result["count_alt"])
    assert result["count"] >= 0
    assert result["count_alt"] >= 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e50_edge():
    """Test edge cases."""
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_50(1, 1)
    assert isinstance(result, dict)
    for key in ("n_picks", "n_types", "count", "count_alt", "forms_agree"):
        assert key in result
    assert result["n_picks"] == 1
    assert result["n_types"] == 1
    assert math.isfinite(result["count"])
    assert math.isfinite(result["count_alt"])
    assert int(result["count"]) == int(result["count_alt"])
    assert result["count"] >= 0
    assert result["count_alt"] >= 0
