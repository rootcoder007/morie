"""Tests for morie.fn.lordzs: Lord's chi-square recomputed."""

import math

import pytest

from morie.fn.lordzs import lord_chi_square


def test_two_parameter_statistic():
    d = [1.2 - 0.9, 0.3 - 0.1]
    S = [[0.05, 0.01], [0.01, 0.04]]
    det = S[0][0] * S[1][1] - S[0][1] ** 2
    stat = (S[1][1] * d[0] ** 2 - 2 * S[0][1] * d[0] * d[1] + S[0][0] * d[1] ** 2) / det
    r = lord_chi_square([1.2, 0.3], [0.9, 0.1], [[0.02, 0.004], [0.004, 0.015]], [[0.03, 0.006], [0.006, 0.025]])
    assert abs(r["statistic"] - stat) < 1e-12
    assert abs(r["pvalue"] - math.exp(-stat / 2)) < 1e-12


def test_one_parameter_and_summed_covariance():
    r = lord_chi_square(0.8, 0.2, 0.09)
    assert abs(r["statistic"] - 0.36 / 0.09) < 1e-12
    assert abs(r["pvalue"] - math.erfc(math.sqrt(2.0))) < 1e-12
    with pytest.raises(ValueError):
        lord_chi_square([1.0, 2.0], [0.0], [[1.0]])
