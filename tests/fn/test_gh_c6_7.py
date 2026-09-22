"""Tests for gh_c6_7.ghosal_kl_diverge."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_7 import ghosal_kl_diverge


def test_gh_c6_7_basic():
    """Test basic functionality."""
    p0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    result = ghosal_kl_diverge(p0, p)
    assert "estimate" in result
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))

    # K(p0; p) for discrete densities: K(p0; p) = sum_i p0_i * log(p0_i / p_i)
    # Using the formula directly on normalized p0 and p:
    s0 = float(np.sum(p0))
    s1 = float(np.sum(p))
    expected = 0.0
    for i in range(len(p0)):
        q = float(p0[i]) / s0
        pi = float(p[i]) / s1
        if q > 0:
            expected += q * np.log(q / pi)
    assert np.isclose(float(estimate), expected)


def test_gh_c6_7_edge():
    """Test edge cases."""
    p0 = np.array([42.0])
    p = np.array([7.0])
    result = ghosal_kl_diverge(p0, p)
    assert "estimate" in result
    # Single-element same-shape: K(p0; p) = 0 since the distributions are identical after normalization.
    assert np.isclose(float(result["estimate"]), 0.0)
