"""Tests for rform -- parallel forms reliability."""

import pytest

from morie.fn import _array_core as np
from morie.fn._containers import ESRes
from morie.fn.rform import parallel_form_reliability


class TestParallelForms:
    def test_high_corr(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(100)
        b = a + rng.standard_normal(100) * 0.1
        result = parallel_form_reliability(a, b)
        assert isinstance(result, ESRes)
        assert result.estimate > 0.9

    def test_ci_covers_estimate(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(50)
        b = a + rng.standard_normal(50) * 0.5
        result = parallel_form_reliability(a, b)
        assert result.ci_lower <= result.estimate <= result.ci_upper


def test_parallel_forms_r_and_fisher_interval():
    import math

    a = [10.0, 12.0, 9.0, 15.0, 11.0, 14.0]
    b = [11.0, 13.0, 8.0, 14.0, 12.0, 15.0]
    n = 6
    ma, mb = sum(a) / n, sum(b) / n
    r = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(
        sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)
    )
    se = 1 / math.sqrt(n - 3)
    res = parallel_form_reliability(a, b)
    assert res.estimate == pytest.approx(r, rel=1e-12)
    assert res.ci_lower == pytest.approx(math.tanh(math.atanh(r) - 1.96 * se), rel=1e-12)
