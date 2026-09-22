"""Tests for dctgls.doubly_censored_gls."""

from morie.fn import _array_core as np

from morie.fn.dctgls import doubly_censored_gls


def test_dctgls_basic():
    """Test basic functionality with left censoring."""
    rng = np.random.default_rng(42)
    n = 200
    p = 3
    X = rng.normal(size=(n, p))
    beta_true = np.array([1.5, -0.7, 0.3])
    y = X @ beta_true + rng.normal(size=n)
    left = -2.0
    y_obs = np.maximum(y, left)

    result = doubly_censored_gls(y_obs, X, left=left)

    assert isinstance(result, dict)

    # Documented return keys
    for key in ("beta", "se", "weights", "n_left", "n_right",
                "n_complete", "naive_complete_case", "effective_sample_size"):
        assert key in result, "missing key: %s" % key

    beta = np.asarray(result["beta"]).ravel()
    assert beta.shape == (p + 1,), "beta should have shape (p+1,)"

    # Some left-censored observations must be present
    assert result["n_left"] > 0
    # Censoring survival weights must be finite
    assert np.all(np.isfinite(np.asarray(result["weights"])))
    # Effective sample size is Kish's formula
    w = np.asarray(result["weights"])
    ess_expected = float(w.sum() ** 2 / np.sum(w ** 2))
    assert abs(result["effective_sample_size"] - ess_expected) < 1e-10


def test_dctgls_edge():
    """Test edge case: no censoring (delta supplied)."""
    rng = np.random.default_rng(42)
    n = 150
    X = rng.normal(size=(n, 2))
    beta_true = np.array([0.5, -1.0])
    y = X @ beta_true + rng.normal(size=n)
    delta = np.ones(n)

    result = doubly_censored_gls(y, X, delta=delta)

    assert isinstance(result, dict)

    beta = np.asarray(result["beta"]).ravel()
    assert beta.shape == (3,)

    # No censoring observed
    assert result["n_left"] == 0
    assert result["n_right"] == 0
    assert result["n_complete"] == n

    # With delta=1 for everyone, G clipped at trunc and weights ~ 1/trunc,
    # so estimate should be close to OLS on the same data.
    B = np.column_stack([np.ones(n), X])
    beta_ols_expected = np.linalg.solve(B.T @ B, B.T @ y)
    # IPCW with delta=1 is proportional to OLS; check proportionality
    ratio = beta[1:] / beta_ols_expected[1:]
    assert np.all(np.abs(ratio - ratio[0]) < 1e-6)
