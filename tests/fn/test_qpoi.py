"""Tests for morie.fn.qpoi — Poisson quantile function."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.qpoi import qpois


class TestQpois:
    """Tests for qpois()."""

    def test_median_lambda1(self):
        """qpois(0.5, lambda_=1) = 1."""
        result = qpois(0.5, lambda_=1.0)
        assert int(result) == 1

    def test_high_quantile(self):
        """qpois(0.99, lambda_=1) should be a small positive int."""
        result = qpois(0.99, lambda_=1.0)
        assert int(result) >= 1

    def test_type(self):
        """Returns a finite numeric array of length 1."""
        result = qpois(0.5, lambda_=5.0)
        arr = np.asarray(result)
        assert arr.size == 1
        assert math.isfinite(float(arr))

    def test_raises_nonpositive_lambda(self):
        """Should reject lambda_ <= 0."""
        with pytest.raises(ValueError):
            qpois(0.5, lambda_=0.0)


def test_qpois_is_the_smallest_k_with_cdf_at_least_p():
    for p in (0.05, 0.5, 0.95):
        cdf, k = 0.0, -1
        while cdf < p:
            k += 1
            cdf += math.exp(-3.2) * 3.2**k / math.factorial(k)
        assert int(qpois(p, lambda_=3.2)) == k
