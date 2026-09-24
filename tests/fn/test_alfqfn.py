"""Tests for alfqfn.alphazero_q_function."""

from morie.fn import _array_core as np

from morie.fn.alfqfn import alphazero_q_function


def test_alfqfn_basic():
    """Test basic functionality with 1-D N and v."""
    rng = np.random.default_rng(44)
    n_actions = 5
    N = list(rng.integers(1, 20, n_actions))
    v = list(rng.normal(0, 1, n_actions))
    result = alphazero_q_function(N, v)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "q" in result
    assert "w" in result
    assert "n" in result
    assert len(result["q"]) == n_actions
    assert len(result["w"]) == n_actions
    assert len(result["n"]) == n_actions


def test_alfqfn_edge():
    """Test edge case with an unvisited first action."""
    rng = np.random.default_rng(44)
    n_actions = 4
    N = [0, 5, 3, 2]
    v = list(rng.normal(0, 1, n_actions))
    result = alphazero_q_function(N, v, unvisited=1.5)
    assert isinstance(result, dict)
    assert result["estimate"] == 1.5
    assert result["q"][0] == 1.5
