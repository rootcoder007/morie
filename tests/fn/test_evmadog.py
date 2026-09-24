"""Tests for evmadog.evt_madogram."""

import math

from morie.fn import _array_core as np

from morie.fn.evmadog import evt_madogram


def test_evmadog_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    t = 0.5
    result = evt_madogram(x, y, t)
    assert isinstance(result, dict)
    assert len(result) > 0
    for value in result.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)


def test_evmadog_edge():
    """Test edge cases with minimal valid input."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 20)
    y = rng.normal(0, 1, 20)
    t = 0.5
    result = evt_madogram(x, y, t)
    assert isinstance(result, dict)
    for value in result.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)
