"""Tests for esliwls.esl_iwls."""

from morie.fn import _array_core as np

from morie.fn.esliwls import esl_iwls


def test_esliwls_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    p = 5
    X = rng.normal(0, 1, (n, p))
    # Generate 0/1 response consistent with a logistic model.
    true_beta = rng.normal(0, 1, p + 1)  # +1 for intercept added by the function
    eta = true_beta[0] + X @ true_beta[1:]
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < prob).astype(float)
    beta0 = np.zeros(p + 1)
    result = esl_iwls(X, y, beta0)
    # Documented return keys.
    expected_keys = {
        "beta", "se", "z", "p_value", "fitted", "loglik",
        "deviance", "n_iter", "converged", "separated",
    }
    assert isinstance(result, dict)
    for k in expected_keys:
        assert k in result, f"missing key: {k}"
    # Shape checks: X has p columns, function adds intercept -> p+1 betas.
    assert result["beta"].shape == (p + 1,)
    assert result["se"].shape == (p + 1,)
    assert result["z"].shape == (p + 1,)
    assert result["p_value"].shape == (p + 1,)
    assert result["fitted"].shape == (n,)
    # Fitted probabilities live in (0, 1) for a non-separated fit.
    assert np.all(result["fitted"] > 0.0)
    assert np.all(result["fitted"] < 1.0)
    # p-values must lie in [0, 1].
    assert np.all(result["p_value"] >= 0.0)
    assert np.all(result["p_value"] <= 1.0)
    # deviance is the negative log-likelihood rescaled; should be finite.
    assert np.isfinite(result["deviance"])
    assert np.isfinite(result["loglik"])
    # On a non-degenerate problem IRLS should converge.
    assert bool(result["converged"]) is True
    assert bool(result["separated"]) is False


def test_esliwls_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 100
    p = 5
    X = rng.normal(0, 1, (n, p))
    eta = rng.normal(0, 1, n)
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < prob).astype(float)
    result = esl_iwls(X, y)
    assert isinstance(result, dict)
    # Should produce a fitted array of length n.
    assert result["fitted"].shape == (n,)
    # Default starting values are zeros of length p+1 (intercept added).
    assert result["beta"].shape == (p + 1,)
