"""Tests for ca5e9.ca_chapter_5_equation_9."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.ca5e9 import ca_chapter_5_equation_9


def test_ca5e9_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p = 3
    bs = rng.normal(0, 1, p)
    xs = rng.normal(0, 1, p)
    tau_m = 0.5
    result = ca_chapter_5_equation_9(tau_m, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_ca5e9_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    p = 2
    bs = rng.normal(0, 1, p)
    xs = rng.normal(0, 1, p)
    tau_m = 0.0
    result = ca_chapter_5_equation_9(tau_m, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
