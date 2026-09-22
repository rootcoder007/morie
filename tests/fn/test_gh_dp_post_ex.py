"""Tests for gh_dp_post_ex.ghosal_dp_posterior_exact."""

from morie.fn import _array_core as np

from morie.fn.gh_dp_post_ex import ghosal_dp_posterior_exact


def test_gh_dp_post_ex_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dp_posterior_exact(x)
    assert "grid" in result
    assert "base_density" in result
    assert "atoms" in result
    assert "atom_probs" in result
    assert "base_weight" in result
    assert "atom_weight" in result
    assert "n_distinct" in result
    assert "is_density" in result
    assert "limit_note" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 5
    assert result["n_distinct"] == 5
    assert result["is_density"] is False
    assert np.all(np.isfinite(np.asarray(result["base_density"], dtype=float)))
    # Independent computation per docstring formula:
    # total atom mass = n/(alpha+n), with alpha=1.0, n=5 -> 5/6
    n = 5
    alpha = 1.0
    expected_atom_weight = n / (alpha + n)
    expected_base_weight = alpha / (alpha + n)
    assert abs(result["atom_weight"] - expected_atom_weight) < 1e-12
    assert abs(result["base_weight"] - expected_base_weight) < 1e-12
    # Each distinct observation appears once, so per-atom mass = atom_weight / n_distinct
    expected_per_atom = expected_atom_weight / result["n_distinct"]
    assert np.allclose(np.asarray(result["atom_probs"]), expected_per_atom)
    # atoms should match the sorted distinct observations
    assert np.allclose(np.asarray(result["atoms"]), np.sort(np.asarray(x, dtype=float)))


def test_gh_dp_post_ex_edge():
    """Test edge cases."""
    result = ghosal_dp_posterior_exact(np.array([42.0]))
    assert result["n"] == 1
    assert result["n_distinct"] == 1
    assert result["is_density"] is False
    # Independent computation: n=1, alpha=1.0 -> atom_weight 1/2, base_weight 1/2
    n = 1
    alpha = 1.0
    assert abs(result["atom_weight"] - n / (alpha + n)) < 1e-12
    assert abs(result["base_weight"] - alpha / (alpha + n)) < 1e-12
