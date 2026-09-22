"""Tests for ddpest.dependent_dp."""

from morie.fn import _array_core as np

from morie.fn.ddpest import dependent_dp


def _gaussian_atom(x, h):
    """Atom location for covariate x and component index h."""
    return x + h


def test_ddpest_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x_grid = rng.normal(0, 1, 100)
    alpha = 0.05
    K = 5
    atom_fn = _gaussian_atom
    result = dependent_dp(x_grid, alpha, K, atom_fn, rng=rng, seed=0)
    assert isinstance(result, dict)
    assert "G" in result
    assert result["kind"] == "single_weights"
    assert result["K"] == int(K)
    assert "weights" in result
    assert "note" in result
    # Shared membership across x
    shared_w = result["weights"]
    assert isinstance(shared_w, list)
    assert len(shared_w) == int(K)
    # Each covariate has the same shared weights but its own atoms
    G = result["G"]
    assert len(G) == len(x_grid)
    for i, x in enumerate(x_grid):
        key = x
        assert key in G
        per_x = G[key]
        assert per_x["weights"] == shared_w
        assert per_x["atoms"] == [atom_fn(x, h) for h in range(int(K))]


def test_ddpest_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x_grid = rng.normal(0, 1, 100)
    alpha = 0.05
    K = 5
    atom_fn = _gaussian_atom
    result = dependent_dp(x_grid, alpha, K, atom_fn, rng=rng, seed=0)
    assert isinstance(result, dict)
    assert "G" in result
    assert result["kind"] == "single_weights"
