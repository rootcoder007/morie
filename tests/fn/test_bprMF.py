"""Tests for bprMF.bpr_mf."""

from morie.fn import _array_core as np

from morie.fn.bprMF import bpr_mf


def test_bprMF_basic():
    """Test basic functionality."""
    rng_pos = np.random.default_rng(42)
    pairs = {u: rng_pos.integers(0, 5, size=4).tolist() for u in range(3)}
    result = bpr_mf(pairs, 3, 5, k_dim=2, iters=50, seed=0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "W" in result
    assert "H" in result
    assert "k" in result
    assert "bpr_opt_history" in result
    assert "final_bpr_opt" in result
    assert "auc" in result
    assert "param_norm" in result
    assert "regularizer_sign" in result
    assert "method" in result
    assert "caveat" in result
    assert result["k"] == 2
    assert len(result["W"]) == 3
    assert len(result["H"]) == 5
    assert all(len(row) == 2 for row in result["W"])
    assert all(len(row) == 2 for row in result["H"])
    assert result["regularizer_sign"] == "correct"
    assert isinstance(result["bpr_opt_history"], list)
    assert len(result["bpr_opt_history"]) > 0
    assert result["final_bpr_opt"] == result["bpr_opt_history"][-1]
    # Independent computation of the parameter norm from the returned W, H.
    W, H = result["W"], result["H"]
    expected_norm = (sum(v * v for row in W for v in row)
                     + sum(v * v for row in H for v in row)) ** 0.5
    assert abs(result["param_norm"] - expected_norm) < 1e-9


def test_bprMF_edge():
    """Test edge cases."""
    rng_pos = np.random.default_rng(42)
    pairs = {0: [1, 2]}
    result = bpr_mf(pairs, 1, 3, k_dim=2, iters=20, seed=1)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert len(result["W"]) == 1
    assert len(result["H"]) == 3
    assert result["k"] == 2
