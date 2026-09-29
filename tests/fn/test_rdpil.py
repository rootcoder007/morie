"""Statistics is the grammar of science. — Karl Pearson"""

import pytest

from morie.fn import _array_core as np
from morie.fn._containers import TestResult
from morie.fn.rdpil import rdpil, red_pill_test


class TestRdpil:
    def test_alias(self):
        assert rdpil is red_pill_test

    def test_reject_null(self):
        x = np.array([10, 11, 12, 10, 11, 12, 10, 11, 12, 10], dtype=float)
        result = red_pill_test(x, mu0=0.0)
        assert isinstance(result, TestResult)
        assert result.extra["decision"] == "red_pill"

    def test_fail_to_reject(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 30)
        result = red_pill_test(x, mu0=0.0)
        assert result.extra["decision"] == "blue_pill"


def test_one_sample_t_recomputed():
    import math

    x = [2.1, 3.4, 1.9, 5.6, 2.8]
    n = 5
    m = sum(x) / n
    s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    r = red_pill_test(x, mu0=2.0)
    assert r.statistic == pytest.approx((m - 2.0) / (s / math.sqrt(n)), rel=1e-12)
    two = r.p_value
    assert red_pill_test(x, mu0=2.0, alternative="greater").p_value == pytest.approx(two / 2, rel=1e-12)
