"""Tests for hrzc1.horowitz_censored_regression (CLAD)."""

import numpy as np
import pytest

from morie.fn.hrzc1 import horowitz_censored_regression


def test_hrzc1_recovers_the_slope_under_censoring():
    """CLAD (Powell 1984) stays consistent under left-censoring at 0
    where OLS on the censored data is badly biased toward zero."""
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=(n, 1))
    y_star = 1.0 + 2.0 * x[:, 0] + rng.standard_normal(n)
    y = np.maximum(y_star, 0.0)  # ~30 percent censored
    r = horowitz_censored_regression(x, y, censor=0.0)
    est = np.asarray(r["estimate"], dtype=float).ravel()
    slope = est[-1]
    assert slope == pytest.approx(2.0, abs=0.3)
    ols = np.linalg.lstsq(np.column_stack([np.ones(n), x]), y, rcond=None)[0][1]
    assert abs(ols - 2.0) > abs(slope - 2.0)  # CLAD beats naive OLS


def test_hrzc1_uncensored_data_reduces_to_median_regression():
    rng = np.random.default_rng(1)
    n = 800
    x = rng.normal(size=(n, 1))
    y = 0.5 + 1.5 * x[:, 0] + rng.laplace(size=n)  # median-friendly noise
    r = horowitz_censored_regression(x, y, censor=-1e9)
    est = np.asarray(r["estimate"], dtype=float).ravel()
    assert est[-1] == pytest.approx(1.5, abs=0.2)
