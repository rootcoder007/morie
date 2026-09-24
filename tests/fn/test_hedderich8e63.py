"""Verification tests for hedderich8e63.

Hedderich, eq (8.63) -- the odds ratio over an interval of the predictor. Every expected value is recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.hedderich8e63 import hedderich_chapter_8_equation_63


def test_odds_ratio_over_an_interval():
    # (8.63): OR = exp(beta1 (b - a))
    for beta1, a, b in ((0.7, 0.0, 1.0), (-0.3, 2.0, 5.0), (1.2, 1.0, 1.0)):
        res = hedderich_chapter_8_equation_63(beta1, a, b)
        assert res["span"] == pytest.approx(b - a, rel=1e-12)
        assert res["logor"] == pytest.approx(beta1 * (b - a), rel=1e-12)
        assert res["or"] == pytest.approx(math.exp(beta1 * (b - a)), rel=1e-12)


def test_unit_change_gives_exp_of_the_slope():
    assert hedderich_chapter_8_equation_63(0.5)["or"] == pytest.approx(math.exp(0.5), rel=1e-12)


def test_wald_interval_scales_with_the_span():
    res = hedderich_chapter_8_equation_63(0.4, 0.0, 2.0, se=0.1, level=0.95)
    assert res["se_logor"] == pytest.approx(2.0 * 0.1, rel=1e-12)
    # the interval is symmetric on the log scale
    assert math.log(res["ci_high"]) - res["logor"] == pytest.approx(
        res["logor"] - math.log(res["ci_low"]), rel=1e-10)


def test_rejects_a_non_positive_standard_error():
    with pytest.raises(ValueError):
        hedderich_chapter_8_equation_63(0.4, se=0.0)
