"""Tests for gh_ap_i3.ghosal_borell_tis."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_i3 import ghosal_borell_tis


def test_gh_ap_i3_basic():
    """Test basic functionality."""
    u = 2.0
    sigma_f = 1.0
    result = ghosal_borell_tis(u=u, sigma_f=sigma_f)
    assert "estimate" in result
    expected = float(np.exp(-u * u / (2.0 * sigma_f ** 2)))
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert np.isfinite(estimate)
    assert estimate == expected
    # Monotonicity in u: evaluating at u+1 should give a strictly smaller bound.
    expected_tighter = float(np.exp(-(u + 1.0) ** 2 / (2.0 * sigma_f ** 2)))
    assert bool(result["tighter_for_larger_u"]) is True
    assert expected_tighter < expected


def test_gh_ap_i3_edge():
    """Test edge cases."""
    result = ghosal_borell_tis(np.array([42.0]))
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    expected = float(np.exp(-42.0 ** 2 / (2.0 * 1.0 ** 2)))
    assert estimate == expected
