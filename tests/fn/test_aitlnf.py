"""Tests for aitlnf.logistic_normal_fit."""

from morie.fn import _array_core as np

from morie.fn.aitlnf import logistic_normal_fit


def test_aitlnf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    # X must be strictly positive (compositions); exponentiate the
    # normal draws so every entry is positive, then row-normalise to
    # form valid compositions on the simplex.
    X = np.exp(X)
    row_sums = X.sum(axis=1, keepdims=True)
    X = X / row_sums

    result = logistic_normal_fit(X)

    # The function returns a RichResult (mapping) with these keys.
    assert "mu" in result
    assert "Sigma" in result
    assert "center" in result
    assert "loglik" in result
    assert "n" in result
    assert "D" in result

    # Shapes: mu has D-1 entries, Sigma is (D-1)x(D-1), centre is length D.
    n, D = 100, 5
    assert np.asarray(result["mu"]).shape == (D - 1,)
    assert np.asarray(result["Sigma"]).shape == (D - 1, D - 1)
    assert np.asarray(result["center"]).shape == (D,)
    assert result["n"] == n
    assert result["D"] == D

    # Centre is the fitted composition on the simplex and must lie
    # in the simplex (positive, sums to 1, has the right length).
    cen = np.asarray(result["center"])
    assert (cen > 0).all()
    assert abs(cen.sum() - 1.0) < 1e-10

    # Independent reconstruction of the MLE on the alr coordinates.
    # Y = alr(X); muhat = mean(Y); Sigma_hat = cov(Y, ddof=1)
    Y = np.log(X[:, :-1]) - np.log(X[:, -1:])
    mu_expected = Y.mean(axis=0)
    mu_got = np.asarray(result["mu"])
    assert np.allclose(mu_got, mu_expected, atol=1e-10)

    Yc = Y - mu_expected
    Sigma_expected = (Yc.T @ Yc) / (n - 1)
    Sigma_got = np.asarray(result["Sigma"])
    assert np.allclose(Sigma_got, Sigma_expected, atol=1e-10)

    # Sigma must be symmetric and (for an MLE on a continuous sample)
    # positive-definite.
    assert np.allclose(Sigma_got, Sigma_got.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(Sigma_got)
    assert (eigvals > 0).all()


def test_aitlnf_edge():
    """Test edge cases: ddof controls the divisor."""
    rng = np.random.default_rng(123)
    X = rng.normal(0, 1, (50, 4))
    X = np.exp(X)
    X = X / X.sum(axis=1, keepdims=True)

    # ddof=1 (default): divide by n-1, i.e. the unbiased covariance.
    res1 = logistic_normal_fit(X, ddof=1)
    # ddof=0: divide by n, the MLE.
    res0 = logistic_normal_fit(X, ddof=0)

    Y = np.log(X[:, :-1]) - np.log(X[:, -1:])
    n = 50
    mu_expected = Y.mean(axis=0)
    Yc = Y - mu_expected

    Sigma_ddof1 = (Yc.T @ Yc) / (n - 1)
    Sigma_ddof0 = (Yc.T @ Yc) / n

    assert np.allclose(np.asarray(res1["mu"]), mu_expected, atol=1e-10)
    assert np.allclose(np.asarray(res0["mu"]), mu_expected, atol=1e-10)
    assert np.allclose(np.asarray(res1["Sigma"]), Sigma_ddof1, atol=1e-10)
    assert np.allclose(np.asarray(res0["Sigma"]), Sigma_ddof0, atol=1e-10)

    # Different ddof choices must produce different Sigma when n > 1.
    assert not np.allclose(np.asarray(res1["Sigma"]),
                           np.asarray(res0["Sigma"]))
