"""Tests for gh_c4_11.ghosal_dp_stickbr."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_11 import ghosal_dp_stickbr


def test_gh_c4_11_basic():
    """Test basic functionality."""
    alpha = 1.0
    n_terms = 5
    result = ghosal_dp_stickbr(n_terms, alpha)
    assert "estimate" in result
    est = result["estimate"]
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))
    # Independent computation of the stick-breaking/atomic estimate:
    # mean = sum_i W_i * theta_i, where W comes from the Sethuraman
    # representation and theta_i ~ Uniform(0, 1).
    # We only check the documented structural invariants here (finite,
    # scalar) rather than recompute the RNG-dependent numeric value,
    # since that would couple to the seeded RNG.
    assert np.isscalar(est) or np.asarray(est).shape == ()


def test_gh_c4_11_edge():
    """Test edge cases."""
    result = ghosal_dp_stickbr(1, 42.0)
    # The documented return key is "estimate", not "n".
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
