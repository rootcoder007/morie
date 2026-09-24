"""Tests for bayes_simple.bayes_simple."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.bayes_simple import (
    bayes_simple,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e51_basic():
    """Test basic functionality."""
    result = bayes_simple(
        p_z_given_a=0.8,
        p_a=0.3,
        p_z=0.5,
    )
    assert isinstance(result, dict)
    assert "posterior" in result
    value = result["posterior"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_david_j_morin_probability_for_the_enthusiastic_beginner2e51_edge():
    """Test edge cases."""
    result = bayes_simple(
        p_z_given_a=1.0,
        p_a=0.25,
        p_z=0.5,
    )
    assert isinstance(result, dict)
    assert "posterior" in result
    value = result["posterior"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0
