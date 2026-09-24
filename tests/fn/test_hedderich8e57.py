"""Verification tests for hedderich8e57.

Hedderich, eq (8.57) -- the Wald test per coefficient. Every expected value is recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.hedderich8e57 import hedderich_chapter_8_equation_57


def test_wald_statistic_and_interval_per_coefficient():
    # (8.56)-(8.57): z = beta/se, two-sided normal p, beta +- z_level se
    beta = [0.8, -1.5, 0.05]
    se = [0.2, 0.5, 0.1]
    res = hedderich_chapter_8_equation_57(beta, se, level=0.95)
    for i, (b, s) in enumerate(zip(beta, se)):
        assert res["z"][i] == pytest.approx(b / s, rel=1e-12)
        assert res["ci_low"][i] == pytest.approx(
            b - (res["ci_high"][i] - b), rel=1e-10)
        half = res["ci_high"][i] - b
        assert half > 0.0
        # the same critical value scales every standard error
        assert half / s == pytest.approx(
            (res["ci_high"][0] - beta[0]) / se[0], rel=1e-12)


def test_wald_p_value_is_the_two_sided_normal_tail():
    # at exactly the 5 per cent critical value the tail is 0.05 to within
    # 1e-17, and the strict p < alpha comparison therefore does not reject
    res = hedderich_chapter_8_equation_57([1.959963984540054], [1.0], level=0.95)
    exact = 2.0 * (0.5 * math.erfc(1.959963984540054 / math.sqrt(2.0)))
    assert res["pvalue"][0] == pytest.approx(exact, abs=1e-15)
    assert res["reject"][0] is False
    clear = hedderich_chapter_8_equation_57([3.0], [1.0])
    assert clear["pvalue"][0] == pytest.approx(
        2.0 * (0.5 * math.erfc(3.0 / math.sqrt(2.0))), abs=1e-12)
    assert clear["reject"][0] is True
    flat = hedderich_chapter_8_equation_57([0.0], [1.0])
    assert flat["pvalue"][0] == pytest.approx(1.0, abs=1e-12)
    assert flat["reject"][0] is False


def test_wald_rejects_a_non_positive_standard_error():
    with pytest.raises(ValueError):
        hedderich_chapter_8_equation_57([1.0], [0.0])
    with pytest.raises(ValueError):
        hedderich_chapter_8_equation_57([1.0, 2.0], [1.0])
