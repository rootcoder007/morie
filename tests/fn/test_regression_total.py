"""Tests for morie.fn.regression_total.

Brus, D. J. (2022). Spatial Sampling with R, eq. (9.2).
Inputs are chosen so the expected value is exact by hand:
  t_regr = t_pi(z) + b (t(x) - t_pi(x)) = 100 + 2*(50-30) = 140
"""

import pytest

from morie.fn.regression_total import regression_total


def test_regression_total_matches_the_book_equation():
    r = regression_total(100.0, 50.0, 30.0, 2.0)
    assert r["value"] == pytest.approx(140.0, abs=1e-12)


def test_regression_total_reduces_to_the_pi_estimator_without_bias():
    # when the auxiliary total is already estimated correctly the
    # correction term is zero, whatever the slope
    r = regression_total(100.0, 40.0, 40.0, 7.5)
    assert r["value"] == pytest.approx(100.0, abs=1e-12)
