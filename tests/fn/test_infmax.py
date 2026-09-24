"""Tests for infmax.infomax_objective."""

from morie.fn import _array_core as np

from morie.fn.infmax import infomax_objective


def test_infmax_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    T_network = rng.normal(0, 1, (40, 3))
    def critic(x, y):
        return 0.0
    result = infomax_objective(X, T_network, critic=critic)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_infmax_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    X = rng.normal(0, 1, (10, 2))
    T_network = rng.normal(0, 1, (10, 2))
    def critic(x, y):
        return 0.0
    result = infomax_objective(X, T_network, critic=critic)
    assert isinstance(result, dict)
