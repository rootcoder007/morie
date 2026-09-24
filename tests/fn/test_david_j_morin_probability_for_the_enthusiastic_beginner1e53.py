"""Tests for david_j_morin_probability_for_the_enthusiastic_beginner1e53.david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_53."""

import pytest

from morie.fn import _array_core as np

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner1e53 import (
    david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_53,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e53_basic():
    """Test basic functionality."""
    n = 6
    N = 10
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_53(n, N)
    assert isinstance(result, dict)
    assert "count" in result
    assert "n_picks" in result
    assert "n_types" in result
    assert "count_alt" in result
    assert "forms_agree" in result
    assert result["n_picks"] == n
    assert result["n_types"] == N
    assert int(result["count"]) > 0
    assert int(result["count"]) == int(result["count_alt"])
    assert bool(result["forms_agree"]) is True


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e53_edge():
    """Test edge cases."""
    # n = 0: there is exactly one way to distribute zero indistinguishable balls
    # into N boxes (each box gets zero), so the count is 1.
    result = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_53(0, 10)
    assert isinstance(result, dict)
    assert "count" in result
    assert "n_picks" in result
    assert "n_types" in result
    assert result["n_picks"] == 0
    assert result["n_types"] == 10
    assert int(result["count"]) == 1
    assert int(result["count"]) == int(result["count_alt"])
    assert bool(result["forms_agree"]) is True
