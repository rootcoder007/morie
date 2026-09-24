"""Tests for ca11e38.ca_chapter_11_equation_38."""

import math

import pytest
from morie.fn import _array_core as np

from morie.fn.ca11e38 import ca_chapter_11_equation_38


def test_ca11e38_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    ys = rng.normal(0, 1, (40,))
    ws = rng.uniform(0.5, 1.5, (40,))
    z_cv = 1.96
    result = ca_chapter_11_equation_38(ys, ws, z_cv)
    assert isinstance(result, dict)
    assert "lower" in result
    assert math.isfinite(result["lower"])


def test_ca11e38_edge():
    """Test edge cases with equal weights and a larger critical value."""
    rng = np.random.default_rng(123)
    ys = rng.normal(5, 2, (40,))
    ws = np.ones(40)
    z_cv = 2.576
    result = ca_chapter_11_equation_38(ys, ws, z_cv)
    assert isinstance(result, dict)
    assert "lower" in result
    assert math.isfinite(result["lower"])
