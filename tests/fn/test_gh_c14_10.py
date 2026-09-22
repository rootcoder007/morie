"""Tests for gh_c14_10.ghosal_py_eppf."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_10 import ghosal_py_eppf


def test_gh_c14_10_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_py_eppf(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value

    # Independent check via the documented formula:
    # p(n_1..n_k) = [prod_{j<k}(theta + j d)] / (theta + 1)^{[n-1]}
    #              * prod_j (1 - d)^{[n_j - 1]}
    import math
    ns = [1, 2, 3, 4, 5]
    d, theta = 0.5, 1.0
    n = sum(ns)
    k = len(ns)
    lp_expected = 0.0
    for j in range(1, k):
        lp_expected += math.log(theta + j * d)
    for i in range(1, n):
        lp_expected -= math.log(theta + i)
    for nj in ns:
        for l in range(nj - 1):
            lp_expected += math.log(1.0 - d + l)
    expected_estimate = math.exp(lp_expected)
    assert abs(float(result["estimate"]) - expected_estimate) < 1e-12


def test_gh_c14_10_edge():
    """Test edge cases."""
    result = ghosal_py_eppf(np.array([42.0]))
    # The function does not return a key "n"; it returns "estimate", "log_eppf", "method".
    assert "estimate" in result
    assert "log_eppf" in result

    # Independent check for a single-block EPPF:
    # With k=1, the leading numerator product is empty (=1), and prod_j (1-d)^{[n_j-1]}
    # contributes (1-d)^{n_1-1} * (1-d+1) * ... * (1-d+n_1-2) for n_1=42.
    import math
    n_1 = 42
    d, theta = 0.5, 1.0
    lp_expected = 0.0
    for i in range(1, n_1):
        lp_expected -= math.log(theta + i)
    for l in range(n_1 - 1):
        lp_expected += math.log(1.0 - d + l)
    expected_estimate = math.exp(lp_expected)
    assert abs(float(result["estimate"]) - expected_estimate) < 1e-12
    assert abs(float(result["log_eppf"]) - lp_expected) < 1e-12
