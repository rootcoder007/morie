"""Tests for gh_c6_6.ghosal_kl_support."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_6 import ghosal_kl_support


def test_gh_c6_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_kl_support(x)
    assert "estimate" in result
    assert "kl_property" in result
    assert "method" in result
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))
    assert 0.0 <= float(estimate) <= 1.0
    # Documented method string must be present.
    assert "KL support" in result["method"]
    # kl_property should be a boolean indicating estimate > 0.
    assert result["kl_property"] == (float(estimate) > 0)


def test_gh_c6_6_edge():
    """Test edge case: a single-cell (k=1) prior must have KL support == 1
    because the only probability vector on a 1-simplex is [1], so the KL
    neighborhood always contains prior mass 1 (estimate == 1.0)."""
    x = np.array([42.0])
    result = ghosal_kl_support(x, n_sim=200)
    # The returned RichResult must expose the documented 'estimate' key,
    # not 'n'.
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    # On the 1-simplex Pi is uniquely [1], so KL(p0 || Pi) = 0 for every p0
    # already in KL form, and the estimated prior mass of the KL neighborhood
    # must be 1.0 (computed independently from the documented definition).
    expected = 1.0
    assert abs(estimate - expected) < 1e-12
    assert result["kl_property"] is True
