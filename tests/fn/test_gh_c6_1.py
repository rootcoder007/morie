"""Tests for gh_c6_1.ghosal_weak_consist."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_1 import ghosal_weak_consist


def test_gh_c6_1_basic():
    """Test basic functionality."""
    result = ghosal_weak_consist(theta0=0.3, eps=0.1, ns=(20, 80, 320, 1280),
                                 seed=42)
    assert "estimate" in result
    assert "mass_outside_by_n" in result
    estimate = result["estimate"]
    masses = result["mass_outside_by_n"]
    assert np.all(np.isfinite(np.asarray(estimate, dtype=float)))
    assert np.all(np.isfinite(np.asarray(masses, dtype=float)))
    assert len(masses) == 4
    # The estimate key should be the last mass
    assert float(np.asarray(estimate)) == float(np.asarray(masses[-1]))


def test_gh_c6_1_edge():
    """Test edge cases with a single n."""
    result = ghosal_weak_consist(theta0=0.3, eps=0.1, ns=(50,), seed=42)
    assert "estimate" in result
    assert "mass_outside_by_n" in result
    assert len(result["mass_outside_by_n"]) == 1
