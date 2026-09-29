"""Tests for morie.fn.qnbm — negative binomial quantile function."""

import pytest

from morie.fn import _array_core as np
from morie.fn.qnbm import qnbm


class TestQnbm:
    """Tests for qnbm()."""

    def test_median(self):
        """qnbm(0.5, size=1, prob=0.5) = 0."""
        result = qnbm(0.5, size=1, prob=0.5)
        assert int(result) == 0

    def test_nonnegative(self):
        """Quantile is non-negative."""
        result = qnbm(0.9, size=3, prob=0.5)
        assert result >= 0

    def test_type(self):
        """Returns integer-like."""
        result = qnbm(0.5, size=2, prob=0.4)
        assert isinstance(result, (int, np.integer, float, np.floating))

    def test_raises_bad_prob(self):
        """Should reject prob not in (0, 1]."""
        with pytest.raises(ValueError):
            qnbm(0.5, size=1, prob=0)


def test_qnbinom_is_the_smallest_k_with_cdf_at_least_p():
    import math

    size, prob = 3, 0.4
    for p in (0.1, 0.5, 0.9):
        cdf, k = 0.0, -1
        while cdf < p:
            k += 1
            cdf += math.comb(k + size - 1, k) * prob**size * (1 - prob) ** k
        assert int(qnbm(p, size=size, prob=prob)) == k
