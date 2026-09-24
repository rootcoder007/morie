"""Verification tests for hedderich8e80.

Hedderich, eq (8.80) -- the Poisson deviance and goodness of fit. Every expected value is recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.hedderich8e80 import hedderich_chapter_8_equation_80


def test_poisson_deviance_and_pearson_statistic():
    # (8.80): D = 2 sum [y log(y/lam) - (y - lam)]
    y = [3.0, 7.0, 0.0, 5.0]
    lam = [4.0, 6.0, 1.0, 5.0]
    res = hedderich_chapter_8_equation_80(y, lam)
    d = 2.0 * sum((yi * math.log(yi / li) if yi > 0 else 0.0) - (yi - li)
                  for yi, li in zip(y, lam))
    assert res["deviance"] == pytest.approx(d, rel=1e-10)
    chi = sum((yi - li) ** 2 / li for yi, li in zip(y, lam))
    assert res["pearson_chisq"] == pytest.approx(chi, rel=1e-10)


def test_a_perfect_fit_has_zero_deviance():
    y = [2.0, 5.0, 9.0]
    res = hedderich_chapter_8_equation_80(y, y)
    assert res["deviance"] == pytest.approx(0.0, abs=1e-12)
    assert res["pearson_chisq"] == pytest.approx(0.0, abs=1e-12)


def test_degrees_of_freedom_and_decision_when_predictors_are_given():
    y = [3.0, 7.0, 4.0, 5.0, 6.0, 8.0]
    lam = [4.0, 6.0, 5.0, 5.0, 6.0, 7.0]
    res = hedderich_chapter_8_equation_80(y, lam, p=1)
    assert res["df"] == len(y) - 1 - 1
    assert 0.0 <= res["pvalue"] <= 1.0
    assert res["reject"] is False


def test_rejects_a_non_positive_fitted_mean():
    with pytest.raises(ValueError):
        hedderich_chapter_8_equation_80([1.0], [0.0])
