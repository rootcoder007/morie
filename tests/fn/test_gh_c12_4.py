"""Tests for gh_c12_4.ghosal_semipara_bvm."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_4 import ghosal_semipara_bvm


def test_gh_c12_4_basic():
    """Test basic functionality against the documented formula."""
    result = ghosal_semipara_bvm(n=2000, alpha=2.0, n_sim=400, seed=42)

    assert "estimate" in result
    assert "efficient_variance" in result
    assert "gap" in result
    assert "method" in result

    # Independent expectation: efficient variance for a uniform(0,1) truth.
    expected_efficient_variance = 1.0 / 12.0
    expected_gap = abs(result["estimate"] - expected_efficient_variance)

    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["efficient_variance"], dtype=float)))
    assert result["efficient_variance"] == expected_efficient_variance
    assert result["gap"] == expected_gap


def test_gh_c12_4_edge():
    """Test edge case with minimal, valid keyword arguments."""
    result = ghosal_semipara_bvm(n=10, alpha=2.0, n_sim=5, seed=42)

    assert "estimate" in result
    assert "efficient_variance" in result
    assert "gap" in result

    expected_efficient_variance = 1.0 / 12.0
    assert result["efficient_variance"] == expected_efficient_variance
    assert result["gap"] == abs(result["estimate"] - expected_efficient_variance)
