"""Tests for morie.fn.bernoulli_variance: recompute Morin (2016) from the formula."""

from morie.fn.bernoulli_variance import bernoulli_variance


def test_pq():
    for p in (0.0, 0.3, 0.5, 0.91):
        assert abs(bernoulli_variance(p)["variance"] - p * (1 - p)) < 1e-15
    # the variance of a Bernoulli is E[X^2] - E[X]^2 = p - p^2
    assert abs(bernoulli_variance(0.3)["variance"] - (0.3 - 0.09)) < 1e-15
