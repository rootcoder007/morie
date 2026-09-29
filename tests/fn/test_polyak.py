"""Tests for polyak.polyak_target."""

from morie.fn import _array_core as np
from morie.fn.polyak import polyak_target


def test_polyak_basic():
    """Test basic functionality."""
    iterates = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = polyak_target(iterates)
    assert isinstance(result, dict)
    assert "average" in result


def test_polyak_edge():
    """Test edge cases."""
    iterates = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = polyak_target(iterates)
    assert isinstance(result, dict)


def test_polyak_average_soft_update_and_halflife():
    import math

    import pytest

    from morie.fn.polyak import lag_halflife, running_average, soft_update

    it = [[1.0, 2.0], [3.0, 0.0], [2.0, 1.0], [4.0, 5.0]]
    r = polyak_target(it, burn_in=1)
    assert r["average"] == pytest.approx([3.0, 2.0], rel=1e-15)
    assert soft_update([1.0, 1.0], [3.0, -1.0], tau=0.25) == pytest.approx([1.5, 0.5], rel=1e-15)
    assert running_average([1.0], [3.0], decay=0.9) == pytest.approx([1.2], rel=1e-15)
    assert lag_halflife(0.01)["halflife"] == pytest.approx(math.log(0.5) / math.log(0.99), rel=1e-14)
