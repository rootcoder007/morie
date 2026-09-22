"""Tests for gh_c14_15.ghosal_ncrm_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_15 import ghosal_ncrm_def


def test_gh_c14_15_basic():
    """Test basic functionality."""
    jump_sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    jump_locs = np.array([0.1, 0.2, 0.3, 0.4, 0.6])
    set_lo, set_hi = 0.0, 0.5
    result = ghosal_ncrm_def(jump_sizes, jump_locs, set_lo=set_lo, set_hi=set_hi)
    assert "estimate" in result
    est = float(result["estimate"])
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))

    # Independent recomputation from the formula:
    # mass = sum_{tau_k in [set_lo, set_hi)} J_k / sum_k J_k
    total = float(np.sum(jump_sizes))
    in_set = float(np.sum(jump_sizes[(jump_locs >= set_lo) & (jump_locs < set_hi)]))
    expected = in_set / total
    assert abs(est - expected) < 1e-12


def test_gh_c14_15_edge():
    """Test edge cases."""
    # Single jump that lies inside the default set [0.0, 0.5): mass must be 1.0
    result = ghosal_ncrm_def(np.array([42.0]), np.array([0.25]))
    assert "estimate" in result
    assert abs(float(result["estimate"]) - 1.0) < 1e-12
    assert result["total_mass"] == 1.0
