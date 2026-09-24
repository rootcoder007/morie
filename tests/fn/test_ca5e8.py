"""Tests for ca5e8.ca_chapter_5_equation_8."""

import math

from morie.fn import _array_core as np

from morie.fn.ca5e8 import ca_chapter_5_equation_8


def test_ca5e8_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    tau_m = 0.5
    bs = [0.3, -0.2]
    xs = rng.normal(0, 1, 2)
    result = ca_chapter_5_equation_8(tau_m, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_ca5e8_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    tau_m = 0.0
    bs = [0.0, 0.0]
    xs = rng.normal(0, 1, 2)
    result = ca_chapter_5_equation_8(tau_m, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isclose(result["value"], tau_m)
