"""Tests for normal_var_ratio_test."""

import pytest

from morie.fn import _array_core as np
from morie.fn.nvrmt import normal_var_ratio_test


class TestNormalVarRatio:
    def test_equal_variances(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        r = normal_var_ratio_test(x, y)
        assert r.test_name == "Variance ratio F-test"
        assert r.p_value > 0.05

    def test_unequal_variances(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 5, 50)
        r = normal_var_ratio_test(x, y)
        assert r.p_value < 0.05

    def test_too_few(self):
        with pytest.raises(ValueError):
            normal_var_ratio_test([1.0], [2.0])


def test_f_ratio_recomputed():
    x = [2.1, 3.4, 1.9, 5.6, 2.8]
    y = [2.0, 2.2, 2.1, 2.6, 2.4, 1.9]

    def v1(a):
        m = sum(a) / len(a)
        return sum((t - m) ** 2 for t in a) / (len(a) - 1)

    r = normal_var_ratio_test(x, y, alternative="greater")
    assert r.statistic == pytest.approx(v1(x) / v1(y), rel=1e-13)
    assert (r.extra["df1"], r.extra["df2"]) == (4, 5)
