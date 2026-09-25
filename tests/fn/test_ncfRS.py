"""Tests for ncfRS.ncf."""

from morie.fn import _array_core as np

from morie.fn.ncfRS import ncf


def test_ncfRS_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(0)
    n_users, n_items = 10, 20
    n_factors = 3
    pos = {}
    for u in range(n_users):
        n_interactions = int(rng.integers(1, 8))
        pos[u] = [int(rng.integers(0, n_items)) for _ in range(n_interactions)]
    result = ncf(pos, n_users, n_items, n_factors)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_ncfRS_edge():
    """Test edge cases."""
    pos = {0: [0, 1], 1: [1, 2]}
    n_users, n_items, n_factors = 2, 3, 1
    result = ncf(pos, n_users, n_items, n_factors)
    assert isinstance(result, dict)
    assert len(result) > 0
