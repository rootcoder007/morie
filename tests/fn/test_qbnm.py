"""Tests for morie.fn.qbnm — binomial quantile function."""

import pytest

from morie.fn import _array_core as np
from morie.fn.qbnm import qbinom


class TestQbinom:
    """Tests for qbinom()."""

    def test_median_fair_coin(self):
        """qbinom(0.5, 10, 0.5) = 5."""
        result = qbinom(0.5, 10, 0.5)
        assert int(result) == 5

    def test_low_quantile(self):
        """qbinom(0.01, 10, 0.5) should be small integer."""
        result = qbinom(0.01, 10, 0.5)
        assert 0 <= int(result) <= 10

    def test_type(self):
        """Returns a numeric type (scalar or array)."""
        result = qbinom(0.5, 10, 0.5)
        assert isinstance(result, (int, np.integer, np.floating, np.ndarray))

    def test_raises_bad_prob(self):
        """Should reject prob outside [0, 1]."""
        with pytest.raises(ValueError):
            qbinom(0.5, 10, 2.0)


def test_qbinom_is_the_smallest_k_with_cdf_at_least_p():
    import math

    for p in (0.05, 0.3, 0.62, 0.97):
        cdf, k = 0.0, -1
        while cdf < p:
            k += 1
            cdf += math.comb(12, k) * 0.35**k * 0.65 ** (12 - k)
        assert int(qbinom(p, 12, 0.35)) == k
    assert int(qbinom(0.2, 12, 0.35, lower_tail=False)) == int(qbinom(0.8, 12, 0.35))
