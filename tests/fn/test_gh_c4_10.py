"""Tests for gh_c4_10.ghosal_dp_polya_urn."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_10 import ghosal_dp_polya_urn


def test_gh_c4_10_basic():
    """Test basic functionality."""
    result = ghosal_dp_polya_urn(n=5, alpha=2.0, seed=42)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert result["n"] == 5
    assert result["method"].startswith("Polya")


def test_gh_c4_10_edge():
    """Test edge cases."""
    result = ghosal_dp_polya_urn(n=1, alpha=0.5, seed=7)
    assert result["n"] == 1
    # independent formula: count distinct values in drawn sequence
    expected_estimate = float(len(set(result["draws_head"])))
    assert result["estimate"] == expected_estimate
