"""Tests for gh_c2_1.ghosal_random_basis_expansion."""

from morie.fn import _array_core as np

from morie.fn.gh_c2_1 import ghosal_random_basis_expansion


def test_gh_c2_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_random_basis_expansion(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c2_1_edge():
    """Test edge cases."""
    result = ghosal_random_basis_expansion(np.array([42.0]))
    # Single-point input: the returned series f has length 1.
    assert len(result["f"]) == 1
    # With one element the estimate equals that element.
    assert result["estimate"] == result["f"][0]
    # Coefficients are populated with exactly n_terms entries.
    assert len(result["coefficients"]) == 12
