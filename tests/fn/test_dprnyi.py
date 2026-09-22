"""Tests for dprnyi.renyi_dp_composition."""

from morie.fn import _array_core as np

from morie.fn.dprnyi import renyi_dp_composition


def test_dprnyi_basic():
    """Test basic functionality."""
    epsilons = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = renyi_dp_composition(epsilons)
    assert isinstance(result, dict)
    assert "rdp_total" in result
def test_dprnyi_edge():
    """Test edge cases."""
    epsilons = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = renyi_dp_composition(epsilons)
    assert isinstance(result, dict)
