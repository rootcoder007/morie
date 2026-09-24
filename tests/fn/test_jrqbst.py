"""Verification tests for jrqbst.jarque_bera.

JB = n/6 (S^2 + (K - 3)^2/4) with S the sample skewness and K the
kurtosis, referred to chi-squared on two degrees of freedom, whose
survival function is exp(-JB/2). Both are recomputed here.
"""

import math

import pytest

from morie.fn.jrqbst import jarque_bera


def _moments(x):
    n = len(x)
    m = sum(x) / n
    m2 = sum((v - m) ** 2 for v in x) / n
    m3 = sum((v - m) ** 3 for v in x) / n
    m4 = sum((v - m) ** 4 for v in x) / n
    return n, m2, m3, m4


def test_statistic_matches_the_skewness_and_kurtosis_formula():
    x = [0.2, -1.1, 0.5, 2.3, -0.7, 0.1, 1.4, -2.0, 0.3, 0.9]
    n, m2, m3, m4 = _moments(x)
    S = m3 / m2 ** 1.5
    K = m4 / m2 ** 2
    jb = n / 6.0 * (S ** 2 + (K - 3.0) ** 2 / 4.0)
    res = jarque_bera(x)
    assert float(res.statistic) == pytest.approx(jb, rel=1e-10)


def test_p_value_is_the_chi_squared_tail_on_two_degrees_of_freedom():
    x = [0.2, -1.1, 0.5, 2.3, -0.7, 0.1, 1.4, -2.0, 0.3, 0.9]
    res = jarque_bera(x)
    assert float(res.p_value) == pytest.approx(
        math.exp(-float(res.statistic) / 2.0), rel=1e-9)


def test_a_symmetric_mesokurtic_sample_is_not_rejected():
    # symmetric about zero, so the skewness term vanishes
    x = [-2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0]
    res = jarque_bera(x)
    assert float(res.p_value) > 0.05


def test_a_strongly_skewed_sample_raises_the_statistic():
    flat = [0.0] * 20 + [1.0] * 20
    skewed = [0.0] * 38 + [10.0, 12.0]
    assert float(jarque_bera(skewed).statistic) > float(jarque_bera(flat).statistic)
