"""Verification tests for msm043.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 6, eq. 6.2 p.172, the non-informative prior. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm043 import mvsml_bayesian_regression_eq_6_2


def test_the_prior_density_is_the_reciprocal_of_the_variance():
    # eq 6.2: f(beta, sigma^2) proportional to sigma^-2, and sigma^-2
    # is 1 / sigma2 when the argument is the variance
    res = mvsml_bayesian_regression_eq_6_2(4.0)
    assert res["estimate"] == pytest.approx(1.0 / 4.0, rel=1e-12)
    assert res["density"] == pytest.approx(0.25, rel=1e-12)
    assert res["log_density"] == pytest.approx(-math.log(4.0), rel=1e-12)


def test_the_density_is_uniform_in_the_log_of_the_standard_deviation():
    # p(sigma2) proportional to 1/sigma2 is exactly the Jacobian of a
    # uniform density in log(sigma), which is what the book assumes
    lo = mvsml_bayesian_regression_eq_6_2(1.0)["density"]
    hi = mvsml_bayesian_regression_eq_6_2(10.0)["density"]
    assert hi == pytest.approx(lo / 10.0, rel=1e-12)


def test_the_prior_is_flat_in_the_regression_coefficients():
    a = mvsml_bayesian_regression_eq_6_2(2.0, beta=[0.0, 0.0])["density"]
    b = mvsml_bayesian_regression_eq_6_2(2.0, beta=[100.0, -50.0])["density"]
    assert a == pytest.approx(b, rel=1e-12)


def test_the_prior_is_reported_as_improper():
    assert mvsml_bayesian_regression_eq_6_2(1.0)["proper"] is False


def test_a_non_positive_variance_is_refused():
    with pytest.raises(ValueError):
        mvsml_bayesian_regression_eq_6_2(0.0)
