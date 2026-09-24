"""Tests for mrgdrv.martingale_concentration."""

import math

from morie.fn import _array_core as np
from morie.fn.mrgdrv import martingale_concentration


def test_mrgdrv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    c = np.abs(rng.normal(0, 1, 100))
    t = 2.0
    result = martingale_concentration(c, t)
    assert isinstance(result, dict)
    assert any(
        isinstance(v, (int, float)) and math.isfinite(v) for v in result.values()
    )


def test_mrgdrv_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    c = np.abs(rng.normal(0, 1, 5))
    t = 1.0
    result = martingale_concentration(c, t)
    assert isinstance(result, dict)
    assert any(
        isinstance(v, (int, float)) and math.isfinite(v) for v in result.values()
    )
