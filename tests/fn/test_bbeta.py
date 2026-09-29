"""Tests for morie.fn.bbeta -- Bayesian beta-binomial model."""

from morie.fn import _array_core as np
from morie.fn.bbeta import bayesian_beta_binomial


def test_returns_dict():
    result = bayesian_beta_binomial([5, 10], [10, 20], n_grid=50)
    assert isinstance(result, dict)
    assert "group_means" in result
    assert "hyper_a" in result


def test_group_means_in_01():
    result = bayesian_beta_binomial([3, 7, 5], [10, 10, 10], n_grid=50)
    assert np.all(result["group_means"] >= 0)
    assert np.all(result["group_means"] <= 1)


def test_ci_contains_mean():
    result = bayesian_beta_binomial([5, 10, 15], [20, 20, 20], n_grid=50)
    for i in range(3):
        assert result["group_ci_lower"][i] <= result["group_means"][i]
        assert result["group_means"][i] <= result["group_ci_upper"][i]


def test_shrinkage():
    k = [0, 10]
    n = [10, 10]
    result = bayesian_beta_binomial(k, n, n_grid=50)
    assert result["group_means"][0] > 0.0
    assert result["group_means"][1] < 1.0


def test_mismatched_lengths():
    try:
        bayesian_beta_binomial([1, 2], [10])
        assert False
    except ValueError:
        pass


def test_marginal_ml_grid_and_posterior_means_recomputed():
    """Beta-binomial marginal log-likelihood maximised over the same grid;
    group posterior means (a + k) / (a + b + n)."""
    import math

    import pytest

    k = [3, 7, 1, 5]
    n = [10, 12, 8, 9]
    ng = 15

    def betaln(a, b):
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    grid = [0.01 + (20.0 - 0.01) * i / (ng - 1) for i in range(ng)]
    best = (-math.inf, None, None)
    for a in grid:
        for b in grid:
            ll = sum(
                math.lgamma(ni + 1)
                - math.lgamma(ki + 1)
                - math.lgamma(ni - ki + 1)
                + betaln(ki + a, ni - ki + b)
                - betaln(a, b)
                for ki, ni in zip(k, n)
            )
            if ll > best[0]:
                best = (ll, a, b)
    r = bayesian_beta_binomial(k, n, n_grid=ng)
    assert (r["hyper_a"], r["hyper_b"]) == pytest.approx((best[1], best[2]), rel=1e-12)
    assert r["marginal_log_lik"] == pytest.approx(best[0], rel=1e-10)
    a, b = best[1], best[2]
    want = [(a + ki) / (a + b + ni) for ki, ni in zip(k, n)]
    assert [float(v) for v in r["group_means"]] == pytest.approx(want, rel=1e-12)
