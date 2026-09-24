"""Tests for ca7e10.ca_chapter_7_equation_10."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.ca7e10 import ca_chapter_7_equation_10


def test_ca7e10_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    b = float(rng.normal(0, 1))
    se = float(abs(rng.normal(0, 1)) + 0.1)  # ensure positive standard error
    result = ca_chapter_7_equation_10(b, se)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    # The z-test computes z = b / se_b per the docstring
    assert result["value"] == pytest.approx(b / se)
    assert "method" in result


def test_ca7e10_edge():
    """Test edge cases."""
    # Edge case: coefficient zero yields z = 0
    b = 0.0
    se = 1.0
    result = ca_chapter_7_equation_10(b, se)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] == pytest.approx(0.0)
    assert "method" in result
