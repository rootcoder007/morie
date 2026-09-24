"""Tests for hedges_j.hedges_j."""

import math

from morie.fn import _array_core as np

from morie.fn.hedges_j import hedges_j


def test_ca11e3_basic():
    """Test basic functionality."""
    n1, n2 = 50, 50
    result = hedges_j(n1, n2)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    # J = 1 - 3/(4(n1+n2)-9), for large samples this should be close to 1 and < 1
    assert 0 < result["value"] < 1


def test_ca11e3_edge():
    """Test edge cases."""
    n1, n2 = 2, 2
    result = hedges_j(n1, n2)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] > 0
