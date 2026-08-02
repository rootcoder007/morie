"""Tests for mmreg.mm_regression_estimator (Yohai 1987 MM)."""

from morie.fn import _array_core as np
from morie.fn.mmreg import mm_regression_estimator


def _data():
    rng = np.random.default_rng(42)
    x1 = rng.normal(0.0, 1.0, 40)
    x2 = rng.normal(0.0, 1.0, 40)
    X = np.column_stack([x1, x2])
    y = 2.0 * x1 - 1.0 * x2 + 0.5
    return X, y, x1, x2


def test_mmreg_recovers_exact_linear_fit():
    X, y, _, _ = _data()
    r = mm_regression_estimator(X, y, n_subsets=30, seed=0)
    b = list(r["beta"])
    assert abs(b[0] - 0.5) < 1e-6      # intercept
    assert abs(b[1] - 2.0) < 1e-6
    assert abs(b[2] + 1.0) < 1e-6


def test_mmreg_resists_gross_outliers():
    X, y, _, _ = _data()
    y = np.asarray(y).copy()
    for i in (0, 1, 2, 3):             # 10% wild contamination
        y[i] = 500.0
    r = mm_regression_estimator(X, y, n_subsets=50, seed=1)
    b = list(r["beta"])
    assert abs(b[1] - 2.0) < 0.2       # robust slope survives
    assert abs(b[2] + 1.0) < 0.2
    assert r["breakdown"] >= 0.4       # high-breakdown estimator
