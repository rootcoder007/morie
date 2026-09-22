"""Tests for eslpcr.esl_pcr."""

from morie.fn import _array_core as np

from morie.fn.eslpcr import esl_pcr


def test_eslpcr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    M = 3
    result = esl_pcr(X, y, M)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "beta" in result
    assert "intercept" in result
    assert "variance_explained" in result
    assert "singular_values" in result
    assert "M" in result
    assert "n" in result
    assert "p" in result
    assert "method" in result
    assert result["M"] == 3
    assert result["n"] == 100
    assert result["p"] == 5
    assert result["method"] == "PCR on centred X; directions chosen by X-variance only"
    assert isinstance(result["beta"], list)
    assert len(result["beta"]) == 5
    assert isinstance(result["singular_values"], list)
    assert len(result["singular_values"]) == 5
    assert 0.0 <= result["variance_explained"] <= 1.0
    assert result["estimate"] == result["beta"][0]
    # Verify PCR formula manually
    Xc = X - X.mean(axis=0)
    ybar = float(y.mean())
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    beta_manual = np.zeros(5)
    for m in range(M):
        z = Xc @ V[:, m]
        zz = float(z @ z)
        beta_manual += (float(z @ (y - ybar)) / zz) * V[:, m]
    assert np.allclose(result["beta"], beta_manual)
    # Verify intercept
    xbar = X.mean(axis=0)
    expected_intercept = ybar - float(xbar @ beta_manual)
    assert result["intercept"] == expected_intercept
    # Verify variance explained
    total = float(np.sum(S ** 2))
    expected_ve = float(np.sum(S[:M] ** 2) / total)
    assert result["variance_explained"] == expected_ve


def test_eslpcr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    # PCR with M = 1 (minimum valid)
    result1 = esl_pcr(X, y, 1)
    assert isinstance(result1, dict)
    assert result1["M"] == 1
    assert 0.0 <= result1["variance_explained"] <= 1.0
    assert len(result1["beta"]) == 5
    assert len(result1["singular_values"]) == 5
    assert result1["estimate"] == result1["beta"][0]
    # PCR with M = min(n-1, p) = 5
    result_max = esl_pcr(X, y, 5)
    assert isinstance(result_max, dict)
    assert result_max["M"] == 5
    assert 0.0 <= result_max["variance_explained"] <= 1.0
    assert len(result_max["beta"]) == 5
    # With M = p, PCR should reproduce OLS on centred design
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    ols = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    assert np.allclose(result_max["beta"], ols)
