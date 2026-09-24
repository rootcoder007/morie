"""Verification tests for hedderich8e67.

Hedderich, eq (8.67) -- the Pearson residual. Every expected value is recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.hedderich8e67 import hedderich_chapter_8_equation_67


def test_pearson_and_deviance_residuals_for_grouped_binomial_data():
    # (8.67)-(8.68)
    y = [3.0, 7.0, 2.0]
    n = [10.0, 10.0, 5.0]
    pi = [0.4, 0.6, 0.5]
    res = hedderich_chapter_8_equation_67(y, pi, n=n)
    for i in range(3):
        mu = n[i] * pi[i]
        pearson = (y[i] - mu) / math.sqrt(mu * (1.0 - pi[i]))
        assert res["pearson"][i] == pytest.approx(pearson, rel=1e-10)
        term = y[i] * math.log(y[i] / mu) + (n[i] - y[i]) * math.log(
            (n[i] - y[i]) / (n[i] - mu))
        dev = math.copysign(math.sqrt(2.0 * term), y[i] - mu)
        assert res["deviance"][i] == pytest.approx(dev, rel=1e-10)
    assert res["D"] == pytest.approx(sum(d ** 2 for d in res["deviance"]), rel=1e-10)


def test_a_perfect_fit_has_zero_residuals():
    y = [4.0, 6.0]
    n = [10.0, 10.0]
    pi = [0.4, 0.6]
    res = hedderich_chapter_8_equation_67(y, pi, n=n)
    for d in res["pearson"]:
        assert d == pytest.approx(0.0, abs=1e-12)
    assert res["D"] == pytest.approx(0.0, abs=1e-12)


def test_influence_is_the_hosmer_lemeshow_deviance_change():
    """(8.69) is the one-step deviance change d^2 + h r_p^2/(1 - h).

    Not the naive d^2/(1 - h): the leverage enters through the Pearson
    residual, which is what makes it the change in deviance from
    deleting the observation.
    """
    y = [3.0, 7.0]
    n = [10.0, 10.0]
    pi = [0.4, 0.6]
    hat = [0.2, 0.5]
    res = hedderich_chapter_8_equation_67(y, pi, n=n, hat=hat)
    for i in range(2):
        mu = n[i] * pi[i]
        r_p = (y[i] - mu) / math.sqrt(mu * (1.0 - pi[i]))
        expected = res["deviance"][i] ** 2 + hat[i] * r_p ** 2 / (1.0 - hat[i])
        assert res["delta_d"][i] == pytest.approx(expected, rel=1e-10)


def test_influence_exceeds_the_plain_squared_deviance():
    y = [3.0, 7.0]
    res = hedderich_chapter_8_equation_67(y, [0.4, 0.6], n=[10.0, 10.0], hat=[0.2, 0.5])
    for i in range(2):
        assert res["delta_d"][i] > res["deviance"][i] ** 2


def test_rejects_a_fitted_probability_on_the_boundary():
    with pytest.raises(ValueError):
        hedderich_chapter_8_equation_67([1.0], [1.0])
