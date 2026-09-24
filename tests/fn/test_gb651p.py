"""Tests for gb651p.gibbons_ctrl_median_power."""

import math
from morie.fn.gb651p import gibbons_ctrl_median_power


def test_gb651p_basic():
    """Test basic functionality."""
    n = 100
    p = 5
    alpha = 0.05
    h = lambda v: math.exp(-v**2 / 2) / math.sqrt(2 * math.pi)
    result = gibbons_ctrl_median_power(n, p, alpha, h)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_gb651p_edge():
    """Test edge cases."""
    n = 40
    p = 3
    alpha = 0.05
    h = lambda v: math.exp(-v**2 / 2) / math.sqrt(2 * math.pi)
    result = gibbons_ctrl_median_power(n, p, alpha, h)
    assert isinstance(result, dict)
    assert len(result) > 0
