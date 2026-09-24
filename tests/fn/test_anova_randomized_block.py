"""Tests for anova_randomized_block.anova_randomized_block."""

import math

from morie.fn import _array_core as np

from morie.fn.anova_randomized_block import anova_randomized_block


def test_ca9e13_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    y = rng.normal(0, 1, n)
    treatment = rng.integers(0, 3, n)
    block = rng.integers(0, 5, n)
    result = anova_randomized_block(y, treatment, block)
    assert isinstance(result, dict)
    assert "f_treatment" in result
    assert math.isfinite(result["f_treatment"])
    assert result["f_treatment"] >= 0


def test_ca9e13_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    n = 24
    y = rng.normal(0, 1, n)
    treatment = rng.integers(0, 2, n)
    block = rng.integers(0, 4, n)
    result = anova_randomized_block(y, treatment, block)
    assert isinstance(result, dict)
    assert "f_treatment" in result
    assert math.isfinite(result["f_treatment"])
    assert result["f_treatment"] >= 0
