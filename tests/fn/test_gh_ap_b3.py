"""Tests for gh_ap_b3.ghosal_renyi_div."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_ap_b3 import ghosal_renyi_div


def test_gh_ap_b3_basic():
    """Test basic functionality."""
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    alpha = 0.5
    result = ghosal_renyi_div(p, q, alpha=alpha)
    assert "estimate" in result

    # Independent computation of the documented formula
    # D_alpha(P||Q) = (alpha - 1)^{-1} * log( sum_i p_i^alpha * q_i^{1-alpha} )
    # with weights normalized (as the implementation does).
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    pn = p_arr / float(np.sum(p_arr))
    qn = q_arr / float(np.sum(q_arr))
    rho = float(np.sum(pn ** alpha * qn ** (1.0 - alpha)))
    expected = math.log(rho) / (alpha - 1.0)

    est = float(np.asarray(result["estimate"], dtype=float))
    assert math.isfinite(est)
    assert abs(est - expected) < 1e-12

    # At alpha = 1/2 the link-gap key must be present and ~0 by construction.
    assert result["hellinger_link_gap"] is not None
    assert abs(float(result["hellinger_link_gap"])) < 1e-12


def test_gh_ap_b3_edge():
    """Test edge cases."""
    result = ghosal_renyi_div(np.array([42.0]), np.array([1.0]))
    # Documented return key is "estimate", not "n".
    assert "estimate" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert math.isfinite(est)
