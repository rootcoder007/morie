"""Tests for eslsvm.esl_svm_kernel."""

import math

from morie.fn import _array_core as np
from morie.fn.eslsvm import esl_svm_kernel


def test_eslsvm_basic():
    """Test basic SMO fit on two well-separated Gaussian clouds."""
    rng = np.random.default_rng(0)
    n_per = 20
    p = 3
    X1 = rng.normal(-2.0, 1.0, (n_per, p))
    X2 = rng.normal(2.0, 1.0, (n_per, p))
    X = list(X1) + list(X2)
    y = [-1] * n_per + [1] * n_per

    result = esl_svm_kernel(X, y, C=1.0, kernel="rbf", seed=1)

    assert isinstance(result, dict)
    for key in ("alpha", "b", "support_", "n_support",
                "decision", "class_", "accuracy", "dual_gap_check"):
        assert key in result

    # alpha is bounded by the box constraint C
    assert max(result["alpha"]) <= 1.0 + 1e-9
    assert min(result["alpha"]) >= -1e-9
    # accuracy is a probability in [0, 1]
    assert 0.0 <= result["accuracy"] <= 1.0
    # equality constraint sum_i alpha_i y_i = 0
    assert abs(result["dual_gap_check"]) < 1e-9


def test_eslsvm_edge():
    """Test held-out newdata evaluated by a linear kernel."""
    rng = np.random.default_rng(1)
    n_per = 20
    p = 3
    X1 = rng.normal(-2.0, 1.0, (n_per, p))
    X2 = rng.normal(2.0, 1.0, (n_per, p))
    X = list(X1) + list(X2)
    y = [-1] * n_per + [1] * n_per

    n_new = 8
    nd1 = rng.normal(-2.0, 1.0, (n_new, p))
    nd2 = rng.normal(2.0, 1.0, (n_new, p))
    newdata = list(nd1) + list(nd2)

    result = esl_svm_kernel(X, y, C=1.0, kernel="linear",
                            newdata=newdata, seed=2)

    assert isinstance(result, dict)
    # decision and class_ are evaluated at the new points
    assert len(result["decision"]) == 2 * n_new
    assert len(result["class_"]) == 2 * n_new
    assert all(math.isfinite(v) for v in result["decision"])
    assert set(result["class_"]).issubset({-1, 1})
