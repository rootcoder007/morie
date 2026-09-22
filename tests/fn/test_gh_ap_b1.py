"""Tests for gh_ap_b1.ghosal_kl_props."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_b1 import ghosal_kl_props


def test_gh_ap_b1_basic():
    """Test basic functionality."""
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = np.array([2.0, 2.0, 2.0, 3.0, 4.0])
    result = ghosal_kl_props(p, q)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # KL >= 0 with equality iff P = Q (after normalization)
    assert result["estimate"] >= -1e-15
    assert result["nonneg"] is True or result["nonneg"] == True
    # d_TV^2 <= KL/2 (Pinsker)
    assert result["pinsker_holds"] is True or result["pinsker_holds"] == True
    # Independent computation of KL from the documented formula:
    #   normalize p and q, then sum a*log(a/b) over a > 0.
    p_sum = float(sum(p))
    q_sum = float(sum(q))
    pn = [float(v) / p_sum for v in p]
    qn = [max(float(v) / q_sum, 1e-300) for v in q]
    expected_kl = sum(a * math.log(a / b) for a, b in zip(pn, qn) if a > 0)
    assert abs(result["estimate"] - expected_kl) < 1e-10
    # Equal distributions give KL = 0.
    res_equal = ghosal_kl_props(np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]))
    assert abs(res_equal["estimate"]) < 1e-14


def test_gh_ap_b1_edge():
    """Test edge cases."""
    result = ghosal_kl_props(np.array([42.0]), np.array([42.0]))
    assert result["estimate"] == 0.0

import math
