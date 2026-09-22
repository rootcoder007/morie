"""Tests for gh_ap_b2.ghosal_kl_variation."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_b2 import ghosal_kl_variation


def test_gh_ap_b2_basic():
    """Test basic functionality."""
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    result = ghosal_kl_variation(p, q)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Independent expectation: V_2 with k=2 (default).
    import math
    p_n = p / np.sum(p)
    q_n = q / np.sum(q)
    expected = sum(
        a * abs(math.log(a / max(b, 1e-300))) ** 2
        for a, b in zip(p_n, q_n) if a > 0
    )
    assert np.isclose(result["estimate"], expected)


def test_gh_ap_b2_edge():
    """Test edge cases."""
    result = ghosal_kl_variation(np.array([42.0]), np.array([1.0]))
    # Single-element distributions: p == q after normalisation => log(1) == 0
    assert np.isclose(result["estimate"], 0.0)
    assert np.isclose(result["kl"], 0.0)
