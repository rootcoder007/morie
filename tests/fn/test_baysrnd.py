"""Tests for baysrnd.shrinkage_random."""

from morie.fn import _array_core as np

from morie.fn.baysrnd import shrinkage_random


def test_baysrnd_basic():
    """Test basic functionality with integer group labels and explicit variances."""
    rng_y = np.random.default_rng(43)
    y = rng_y.normal(0, 1, 30)
    # Group labels must be integer-like; using ints yields 5 groups of size 6.
    group = np.array([i % 5 for i in range(30)], dtype=int)
    # Provide variance components so _pool doesn't need to estimate sigma2
    # from residuals (no X is supplied -> no residual df).
    sigma2 = 1.0
    tau2 = 0.5
    result = shrinkage_random(y, group=group, sigma2=sigma2, tau2=tau2)

    # The function returns a RichResult that supports dict-style access.
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "u_g" in result
    assert "theta" in result
    assert "lambda_g" in result
    assert "mu" in result
    assert "tau2" in result
    assert "sigma2" in result
    assert "n_g" in result
    assert "G" in result

    u_g = result["u_g"]
    theta = result["theta"]
    lam = result["lambda_g"]
    mu = result["mu"]

    # Basic shape checks.
    n_g = result["n_g"]
    G = result["G"]
    assert G == 5
    assert len(u_g) == G
    assert len(theta) == G
    assert len(lam) == G
    assert len(n_g) == G
    assert sum(n_g) == len(y)

    # Numerics computed independently from the documented formula:
    #   u_g = theta_g - mu
    #   theta_g = lambda_g * ybar_g + (1 - lambda_g) * mu
    #   lambda_g = tau2 / (tau2 + sigma2 / n_g)
    #   mu = sum(n_g * theta_nopool) / sum(n_g)
    # First compute the no-pool group means.
    theta_nopool = []
    for gid in range(G):
        idx = [i for i, gi in enumerate(group) if gi == gid]
    n_g_list = [len(idx) for idx in [[i for i, gi in enumerate(group) if gi == gid] for gid in range(G)]]
    for gid in range(G):
        idx = [i for i, gi in enumerate(group) if gi == gid]
        theta_nopool.append(sum(y[i] for i in idx) / n_g_list[gid])

    mu_expected = sum(n_g_list[g] * theta_nopool[g] for g in range(G)) / sum(n_g_list)
    lam_expected = [tau2 / (tau2 + sigma2 / n_g_list[g]) for g in range(G)]
    theta_expected = [
        lam_expected[g] * theta_nopool[g] + (1.0 - lam_expected[g]) * mu_expected
        for g in range(G)
    ]
    u_expected = [theta_expected[g] - mu_expected for g in range(G)]

    assert abs(mu - mu_expected) < 1e-10
    for g in range(G):
        assert abs(theta[g] - theta_expected[g]) < 1e-10
        assert abs(u_g[g] - u_expected[g]) < 1e-10
        assert abs(lam[g] - lam_expected[g]) < 1e-10

    # estimate == u_g[0] per the implementation.
    assert result["estimate"] == u_g[0]


def test_baysrnd_edge():
    """Test edge cases: call via positional X-as-group shim, with explicit variances."""
    rng_y = np.random.default_rng(43)
    y = rng_y.normal(0, 1, 20)
    # Integer group labels yield 4 groups of size 5; passing as X uses the
    # interface-compatibility shim documented in the function.
    X = np.array([i % 4 for i in range(20)], dtype=int)
    result = shrinkage_random(y, X=X, sigma2=1.0, tau2=0.25)
    assert isinstance(result, dict)
    assert "u_g" in result
    assert "theta" in result
    assert "lambda_g" in result
    assert "G" in result
    assert result["G"] == 4
    assert len(result["u_g"]) == 4
