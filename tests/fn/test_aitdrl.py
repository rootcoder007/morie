"""Tests for aitdrl.dirichlet_loglik."""

import math

from morie.fn import _array_core as np

from morie.fn.aitdrl import dirichlet_loglik


def _make_compositions(rng, N, D):
    """Generate N random compositions of dimension D strictly inside the simplex."""
    raw = rng.random((N, D))
    row_sums = raw.sum(axis=1, keepdims=True)
    return raw / row_sums


def test_aitdrl_basic():
    """Test basic functionality with a single alpha vector and multiple compositions."""
    rng = np.random.default_rng(42)
    D = 5
    N = 100

    alpha = np.array([0.05] * D)
    X = _make_compositions(rng, N, D)

    result = dirichlet_loglik(alpha, X)

    assert isinstance(result, dict)
    assert "estimate" in result

    # Independent computation of the literature formula:
    #   l = N * [ln Gamma(sum a) - sum_i ln Gamma(a_i)]
    #       + sum_i (a_i - 1) * sum_n ln x_{n,i}
    a0 = float(sum(alpha))
    ll_expected = N * (
        math.lgamma(a0) - sum(math.lgamma(float(a)) for a in alpha)
    )
    sum_log_x_expected = [
        sum(math.log(float(X[n][i])) for n in range(N)) for i in range(D)
    ]
    for i in range(D):
        ll_expected += (float(alpha[i]) - 1.0) * sum_log_x_expected[i]

    assert math.isclose(float(result["ll"]), ll_expected, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(
        float(result["estimate"]), ll_expected, rel_tol=1e-9, abs_tol=1e-9
    )

    # The sufficient statistic must match what we computed.
    slx = result["sum_log_x"]
    assert len(slx) == D
    for got, want in zip(slx, sum_log_x_expected):
        assert math.isclose(float(got), want, rel_tol=1e-9, abs_tol=1e-9)


def test_aitdrl_edge():
    """Test that a single 1-D composition (one row) is accepted."""
    rng = np.random.default_rng(42)
    D = 3

    alpha = np.array([1.0, 2.0, 3.0])
    x = rng.random(D)
    x = x / x.sum()

    result = dirichlet_loglik(alpha, x)

    assert isinstance(result, dict)
    assert "ll" in result
    assert "score" in result
    assert "sum_log_x" in result

    # Independent computation for a single composition (N = 1).
    a0 = float(sum(alpha))
    ll_expected = (
        math.lgamma(a0)
        - sum(math.lgamma(float(a)) for a in alpha)
    )
    sum_log_x_expected = [math.log(float(x[i])) for i in range(D)]
    for i in range(D):
        ll_expected += (float(alpha[i]) - 1.0) * sum_log_x_expected[i]

    assert math.isclose(
        float(result["ll"]), ll_expected, rel_tol=1e-9, abs_tol=1e-9
    )
    assert len(result["score"]) == D
    assert result["N"] == 1
    assert result["D"] == D
