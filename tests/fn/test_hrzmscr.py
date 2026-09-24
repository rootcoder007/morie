"""Tests for hrzmscr.horowitz_manski_max_score."""

import math

from morie.fn import _array_core as np

from morie.fn.hrzmscr import horowitz_manski_max_score


def test_hrzmscr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    x = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    result = horowitz_manski_max_score(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_hrzmscr_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 20, 2
    x = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    result = horowitz_manski_max_score(x, y)
    assert isinstance(result, dict)
    assert len(result) > 0
