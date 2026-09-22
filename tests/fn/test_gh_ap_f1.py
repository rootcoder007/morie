"""Tests for gh_ap_f1.ghosal_donsker_class."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_f1 import ghosal_donsker_class


def test_gh_ap_f1_basic():
    """Test basic functionality with a Donsker-class value (s > 1/2)."""
    s = 1.0
    result = ghosal_donsker_class(smoothness=s)
    # Independent computation of the documented integral
    # J = int_0^1 eps^{-1/(2s)} deps = 1 / (1 - 1/(2s))
    expected_exponent = 1.0 / (2.0 * s)
    expected_J = 1.0 / (1.0 - expected_exponent)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert result["estimate"] == expected_J
    assert result["donsker"] is True


def test_gh_ap_f1_non_donsker():
    """s <= 1/2 should yield a non-Donsker (infinite) bracketing integral."""
    s = 0.25
    result = ghosal_donsker_class(smoothness=s)
    assert result["donsker"] is False
    assert result["estimate"] == float("inf")


def test_gh_ap_f1_edge():
    """Calling with the default scalar smoothness returns the documented keys."""
    result = ghosal_donsker_class()
    assert "estimate" in result
    assert "donsker" in result
    assert "method" in result
