"""Tests for ca4e16.ca_chapter_4_equation_16."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.ca4e16 import ca_chapter_4_equation_16


def test_ca4e16_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b = float(rng.normal(0, 1))
    se = float(rng.normal(0, 1))
    # Standard error must be positive; take absolute value and add a small offset
    se = abs(se) + 1e-3
    result = ca_chapter_4_equation_16(b, se)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert "method" in result


def test_ca4e16_edge():
    """Test edge cases."""
    b = 0.0
    se = 1.0
    result = ca_chapter_4_equation_16(b, se)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] == 0.0
