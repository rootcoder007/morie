"""Tests for gh_c14_1.ghosal_eppf_def."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c14_1 import ghosal_eppf_def


def test_gh_c14_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_eppf_def(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c14_1_edge():
    """Test edge cases."""
    result = ghosal_eppf_def(np.array([42.0]))
    assert result["estimate"] > 0.0
    assert math.isfinite(result["estimate"])
    assert result["method"] == "EPPF (GvdV 2017 sec. 14.1, eq. 4.20)"


def test_gh_c14_1_symmetry_and_alpha():
    """Test that estimate is symmetric under permutation and depends on alpha."""
    # Symmetry: permuted block sizes yield the same estimate
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ra = ghosal_eppf_def(a)
    rb = ghosal_eppf_def(b)
    assert ra["estimate"] == rb["estimate"]
    assert ra["log_eppf"] == rb["log_eppf"]

    # Alpha dependence with independent computation of the formula:
    # log p = k*log(α) + lgamma(α) + Σ lgamma(n_j) - lgamma(α + n)
    ns = [2, 3, 4, 5]
    n = sum(ns)
    k = len(ns)
    alpha = 2.0
    expected_log = (k * math.log(alpha)
                    + math.lgamma(alpha)
                    + sum(math.lgamma(v) for v in ns)
                    - math.lgamma(alpha + n))
    expected_est = math.exp(expected_log)
    r = ghosal_eppf_def(np.array([float(v) for v in ns]), alpha=alpha)
    assert math.isclose(r["log_eppf"], expected_log, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(r["estimate"], expected_est, rel_tol=1e-12, abs_tol=1e-12)

    # Different alpha on same block sizes must change the estimate
    r_default = ghosal_eppf_def(np.array([2.0, 3.0, 4.0, 5.0]))
    assert r_default["estimate"] != r["estimate"]
