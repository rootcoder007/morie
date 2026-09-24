"""Tests for factorial.factorial."""

import math
import pytest

from morie.fn import _array_core as np

from morie.fn.factorial import (
    factorial,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e1_basic():
    """Test basic functionality."""
    n = 5
    result = factorial(n)
    assert isinstance(result, dict)
    assert "n" in result
    assert "factorial" in result
    assert result["n"] == 5
    assert result["factorial"] == 120
    assert math.isfinite(result["factorial"])


def test_david_j_morin_probability_for_the_enthusiastic_beginner1e1_edge():
    """Test edge cases."""
    n = 0
    result = factorial(n)
    assert isinstance(result, dict)
    assert "n" in result
    assert "factorial" in result
    assert result["n"] == 0
    assert result["factorial"] == 1
    assert math.isfinite(result["factorial"])
