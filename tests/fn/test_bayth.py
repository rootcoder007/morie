"""Tests for bayth.bayes_theorem_genomic."""

import math

from morie.fn import _array_core as np

from morie.fn.bayth import bayes_theorem_genomic


def _uniform_prior(theta):
    # Uniform prior on the grid [0, 1].
    if theta < 0.0 or theta > 1.0:
        return 0.0
    return 1.0


def _normal_likelihood(theta, y):
    # Likelihood of a sample from N(theta, 1), evaluated at theta.
    # L(theta; y) = prod_i exp(-(y_i - theta)^2 / 2) / sqrt(2*pi)
    norm = 1.0 / math.sqrt(2.0 * math.pi)
    log_l = 0.0
    for yi in y:
        log_l += -0.5 * (yi - theta) ** 2
    return (norm ** len(y)) * math.exp(log_l)


def test_bayth_basic():
    """Test basic functionality with a uniform prior and normal likelihood."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayes_theorem_genomic(y, _uniform_prior, _normal_likelihood)
    # The function returns a RichResult; the test's original assertion was a
    # duck-typed membership check, so access via attribute lookup.
    assert hasattr(result, "payload") or hasattr(result, "estimate")

    payload = result.payload if hasattr(result, "payload") else result
    assert "estimate" in payload
    assert "posterior" in payload
    assert "theta" in payload
    assert "marginal" in payload
    assert "post_var" in payload

    # The grid must be the default (0, 1) and have odd length (2001) for Simpson.
    th = list(payload["theta"])
    assert len(th) == 2001
    assert th[0] == 0.0
    assert abs(th[-1] - 1.0) < 1e-12

    # Posterior should integrate (approximately) to 1 over the grid.
    n_grid = len(th)
    h = (1.0 - 0.0) / (n_grid - 1)
    post = list(payload["posterior"])
    s = post[0] + post[n_grid - 1]
    for i in range(1, n_grid - 1):
        s += (4.0 if i % 2 == 1 else 2.0) * post[i]
    integral = s * h / 3.0
    assert abs(integral - 1.0) < 1e-6

    # The posterior mean should be the grid-weighted integral of the
    # normalised posterior (independent expression).
    simpson_weighted = lambda vs: (
        sum((4.0 if i % 2 == 1 else 2.0) * vs[i] for i in range(1, n_grid - 1))
        + vs[0] + vs[n_grid - 1]
    ) * h / 3.0
    mean_from_grid = simpson_weighted([th[i] * post[i] for i in range(n_grid)])
    assert abs(payload["estimate"] - mean_from_grid) < 1e-9
    assert abs(payload["post_mean"] - mean_from_grid) < 1e-9

    # The posterior variance is E[theta^2] - (E[theta])^2 (independent).
    second_moment = simpson_weighted([th[i] * th[i] * post[i] for i in range(n_grid)])
    expected_var = second_moment - mean_from_grid * mean_from_grid
    assert abs(payload["post_var"] - expected_var) < 1e-9

    # n should match the sample size.
    assert payload["n"] == 100


def test_bayth_edge():
    """Test edge cases: custom support and sample size."""
    y = np.random.default_rng(43).normal(0, 1, 50)
    result = bayes_theorem_genomic(
        y, _uniform_prior, _normal_likelihood, grid=(-2.0, 2.0), n_grid=1001
    )
    payload = result.payload if hasattr(result, "payload") else result
    assert "estimate" in payload
    assert "posterior" in payload
    assert "theta" in payload
    assert "marginal" in payload
    assert "post_var" in payload

    th = list(payload["theta"])
    assert len(th) == 1001  # already odd
    assert th[0] == -2.0
    assert abs(th[-1] - 2.0) < 1e-12

    # Posterior integrates to 1 on the requested grid (independent recompute).
    n_grid = len(th)
    h = (2.0 - (-2.0)) / (n_grid - 1)
    post = list(payload["posterior"])
    s = post[0] + post[n_grid - 1]
    for i in range(1, n_grid - 1):
        s += (4.0 if i % 2 == 1 else 2.0) * post[i]
    integral = s * h / 3.0
    assert abs(integral - 1.0) < 1e-6

    # n should match the sample size.
    assert payload["n"] == 50
