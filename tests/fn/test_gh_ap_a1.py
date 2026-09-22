"""Tests for gh_ap_a1.ghosal_weak_conv_def."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_a1 import ghosal_weak_conv_def


def test_gh_ap_a1_basic():
    """Test basic functionality."""
    p_seq_param = (0.4, 0.45, 0.49, 0.499)
    result = ghosal_weak_conv_def(p_seq_param)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_ap_a1_edge():
    """Test edge case: single value sequence."""
    p_seq_param = np.array([0.499])
    p_limit = 0.5
    result = ghosal_weak_conv_def(tuple(p_seq_param.tolist()), p_limit=p_limit)

    # Independent computation per the documented formula:
    # estimate = max over f in {x, cos} of |E_{P_n} f - E_P f|
    # where E_P f = (1-p)*f(0) + p*f(1) and the gap is evaluated at the
    # last element of p_seq_param.
    ps = list(p_seq_param)
    gaps = []
    for f in (lambda x: x, __import__("math").cos):
        lim = (1.0 - p_limit) * f(0.0) + p_limit * f(1.0)
        last_val = (1.0 - ps[-1]) * f(0.0) + ps[-1] * f(1.0)
        gaps.append(abs(last_val - lim))
    expected_estimate = max(gaps)

    assert result["estimate"] == expected_estimate
    assert result["converging"] is True
    assert result["method"] == "weak convergence (GvdV 2017 App A)"
