"""bayreg2: Student-t robust linear regression by EM.

The generated test imported `bayes_robust`, which does not exist.
Rewritten against student_t_regression and anchored on the property that
makes a heavy-tailed likelihood worth having: resistance to an outlier.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.bayreg2 import student_t_regression

X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
Y_CLEAN = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def test_recovers_an_exact_linear_relationship():
    """y = 2x with an intercept term: the slope must come back as 2."""
    r = student_t_regression(X, Y_CLEAN)
    coefs = [float(c) for c in np.asarray(r["coefficients"])]
    assert coefs[-1] == pytest.approx(2.0, abs=1e-4)
    assert r["converged"]


def test_an_outlier_is_downweighted_not_followed():
    """The last point is corrupted. A Student-t fit should keep the slope
    near 2 and give that observation a visibly smaller weight -- this is
    the whole reason to prefer it over least squares.
    """
    y = list(Y_CLEAN)
    y[-1] = 100.0
    r = student_t_regression(X, y, nu=2.0)
    slope = float(np.asarray(r["coefficients"])[-1])
    w = [float(v) for v in np.asarray(r["weights"])]
    assert w[-1] < min(w[:-1])
    assert abs(slope - 2.0) < abs((100.0 - 2.0) / 6.0)


def test_weights_are_positive_and_finite():
    r = student_t_regression(X, Y_CLEAN)
    assert all(0 < float(w) < 1e6 for w in np.asarray(r["weights"]))
