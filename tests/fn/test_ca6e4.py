"""Tests for ca6e4.ca_chapter_6_equation_4."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.ca6e4 import ca_chapter_6_equation_4


def test_ca6e4_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b0 = 0.5
    b1 = -0.2
    x1 = float(rng.normal(0, 1))
    result = ca_chapter_6_equation_4(b0, b1, x1)
    # RichResult is a dict subclass
    assert isinstance(result, dict)
    # The headline key should be present
    assert "value" in result
    value = result["value"]
    # The predicted value should be a finite number
    assert isinstance(value, (int, float))
    assert math.isfinite(value)


def test_ca6e4_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    b0 = 0.0
    b1 = 1.0
    x1 = float(rng.normal(0, 1))
    result = ca_chapter_6_equation_4(b0, b1, x1)
    assert isinstance(result, dict)
    assert "value" in result
    value = result["value"]
    assert isinstance(value, (int, float))
    assert math.isfinite(value)
