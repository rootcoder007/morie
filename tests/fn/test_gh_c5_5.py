"""Tests for gh_c5_5.ghosal_blk_gibbs."""

from morie.fn import _array_core as np

from morie.fn.gh_c5_5 import ghosal_blk_gibbs


def test_gh_c5_5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_blk_gibbs(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c5_5_edge():
    """Test edge cases: single observation must yield one active component."""
    data = np.array([42.0])
    result = ghosal_blk_gibbs(data)
    # The function returns payload keys: estimate, n_active, weights, atoms, method.
    # With n=1 observation, s has a single entry, so |set(s)| == 1.
    assert "estimate" in result
    assert "n_active" in result
    assert result["n_active"] == 1
    # estimate is documented as the same quantity as n_active (active components).
    expected_estimate = float(1)
    assert np.isclose(float(result["estimate"]), expected_estimate)
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # K weights and K atoms are produced by the truncated stick-breaking scheme.
    assert len(result["weights"]) == 10  # default K
    assert len(result["atoms"]) == 10    # default K
