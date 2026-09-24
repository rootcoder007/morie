"""Tests for km122.kamath_ch8_wmd."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.km122 import kamath_ch8_wmd


def test_km122_basic():
    """Test basic functionality with a known optimal solution."""
    # Example from the docstring: optimal cost is 0.5*1 + 0.5*2 = 1.5
    out = kamath_ch8_wmd([0.5, 0.5], [0.5, 0.5],
                         [[1.0, 3.0], [4.0, 2.0]])
    assert isinstance(out, dict)
    assert "estimate" in out
    assert math.isfinite(out["estimate"])
    # The docstring states the exact cost for this instance
    assert round(out["estimate"], 12) == 1.5
    # Flow should be a 2x2 matrix
    assert "flow" in out
    flow = out["flow"]
    assert len(flow) == 2
    assert all(len(row) == 2 for row in flow)


def test_km122_edge():
    """Test that empty weight vectors raise a ValueError."""
    with pytest.raises(ValueError):
        kamath_ch8_wmd([], [0.5, 0.5], [[1.0, 3.0], [4.0, 2.0]])
