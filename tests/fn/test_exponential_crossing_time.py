"""Tests for exponential_crossing_time.exponential_crossing_time."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.exponential_crossing_time import (
    exponential_crossing_time,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e30_basic():
    """Test basic functionality."""
    result = exponential_crossing_time()
    assert isinstance(result, dict)
    assert "t" in result
    assert math.isfinite(result["t"])
    assert result["t"] >= 0


def test_david_j_morin_probability_for_the_enthusiastic_beginner4e30_edge():
    """Test edge cases."""
    result = exponential_crossing_time(rate_fast=0.5, rate_slow=0.1, ratio=2.0)
    assert isinstance(result, dict)
    assert "t" in result
    assert math.isfinite(result["t"])
    assert result["t"] >= 0
