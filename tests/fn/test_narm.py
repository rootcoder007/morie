"""Tests for narm.narm."""

import math
import pytest
from morie.fn import _array_core as np
from morie.fn.narm import narm


def _all_finite(obj):
    """Recursively check that all numeric values in obj are finite."""
    if isinstance(obj, (int, float)):
        return math.isfinite(obj)
    if isinstance(obj, list):
        return all(_all_finite(v) for v in obj)
    return True


def test_narm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    sessions = rng.normal(0, 1, (40, 10))
    K = np.eye(10) + 0.1 * rng.normal(0, 1, (10, 10))
    c_t = rng.normal(0, 1, 10)
    result = narm(sessions, K, c_t)
    assert isinstance(result, dict)
    assert len(result) > 0
    for val in result.values():
        assert _all_finite(val)


def test_narm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    sessions = rng.normal(0, 1, (40, 3))
    K = np.eye(3) + 0.1 * rng.normal(0, 1, (3, 3))
    c_t = np.zeros(3)
    result = narm(sessions, K, c_t)
    assert isinstance(result, dict)
    assert len(result) > 0
    for val in result.values():
        assert _all_finite(val)
