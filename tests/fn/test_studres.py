"""Tests for studres (internally studentized residuals)."""

import numpy as np
import pytest

from morie.fn.studres import studres


def test_studres_matches_a_hand_computed_case():
    """Two symmetric residuals (+1, -1), equal leverage 0.5, n - p = 1:
    s^2 = (1 + 1) / 1 = 2, so r = +/- 1 / sqrt(2 * 0.5) = +/- 1."""
    y = np.array([2.0, 0.0])
    yhat = np.array([1.0, 1.0])
    h = np.array([0.5, 0.5])
    r = studres(y, yhat, h)
    np.testing.assert_allclose(r, [1.0, -1.0], atol=1e-12)


def test_studres_agrees_with_the_regression_definition():
    """Full check against an actual OLS fit: build the hat matrix, fit,
    and compare with the textbook formula computed independently."""
    rng = np.random.default_rng(0)
    n, p = 40, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(size=n)
    H = X @ np.linalg.solve(X.T @ X, X.T)
    yhat = H @ y
    h = np.diag(H)
    out = np.asarray(studres(y, yhat, h), dtype=float)

    resid = y - yhat
    s = np.sqrt(resid @ resid / (n - p))
    want = resid / (s * np.sqrt(1 - h))
    np.testing.assert_allclose(out, want, rtol=1e-10)


def test_studres_high_leverage_inflates_the_residual():
    """The same raw residual studentizes LARGER at higher leverage --
    dividing by sqrt(1 - h) is the whole point."""
    y = np.array([1.0, 1.0, 0.0, 0.0])
    yhat = np.zeros(4)
    lo = np.asarray(studres(y, yhat, np.array([0.1, 0.1, 0.1, 0.1])), dtype=float)
    hi = np.asarray(studres(y, yhat, np.array([0.9, 0.1, 0.1, 0.1])), dtype=float)
    assert abs(hi[0]) > abs(lo[0])
