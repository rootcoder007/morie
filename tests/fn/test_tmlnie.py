"""Tests for tmlnie.tmle_natural_indirect."""

from morie.fn import _array_core as np
from morie.fn.tmlnie import tmle_natural_indirect


def test_tmlnie_basic():
    """Test basic functionality."""
    y11 = np.random.default_rng(42).normal(0, 1, 100)
    y10 = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_natural_indirect(y11, y10)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_tmlnie_edge():
    """Test edge cases."""
    y11 = np.random.default_rng(42).normal(0, 1, 100)
    y10 = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_natural_indirect(y11, y10)
    assert isinstance(result, dict)


def test_nie_plus_nde_is_the_total_effect():
    import pytest

    from morie.fn.tmlnde import tmle_natural_direct

    y11 = [4.0, 5.5, 3.0, 6.0]
    y10 = [3.0, 4.5, 2.0, 5.0]
    y00 = [2.5, 3.0, 2.2, 4.1]
    nie = tmle_natural_indirect(y11, y10)["estimate"]
    nde = tmle_natural_direct(y10, y00)["estimate"]
    assert nie == pytest.approx(1.0, rel=1e-14)
    assert nie + nde == pytest.approx(sum(a - b for a, b in zip(y11, y00)) / 4, rel=1e-13)
