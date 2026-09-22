"""Tests for eslsbt.esl_se_beta."""

from morie.fn import _array_core as np

from morie.fn.eslsbt import esl_se_beta


def test_eslsbt_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    X = np.asarray(rng_x.normal(0, 1, (100, 5)), dtype=float)
    y = np.asarray(rng_y.normal(0, 1, 100), dtype=float)
    n, p = 100, 5
    beta = np.asarray(rng_x.normal(0, 1, p), dtype=float)
    result = esl_se_beta(X, y, beta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "sigma2_hat" in result
    assert "rss" in result
    assert "df_residual" in result
    assert "n" in result
    assert "p" in result
    assert "method" in result
    assert result["n"] == n
    assert result["p"] == p
    assert result["df_residual"] == n - p
    # Independent recomputation of sigma2_hat = RSS / (n - p)
    rss_val = float(np.sum((y - X @ beta) ** 2))
    assert result["rss"] == rss_val
    assert result["sigma2_hat"] == rss_val / (n - p)
    # First se entry must equal sigma2_hat * sqrt(v_00) where v = (X'X)^{-1}
    XtX = X.T @ X
    v = np.linalg.inv(XtX)
    expected_se = np.sqrt(rss_val / (n - p) * v[0, 0])
    assert result["estimate"] == expected_se
    assert result["se"] == [float(np.sqrt(rss_val / (n - p) * v[j, j]))
                            for j in range(p)]


def test_eslsbt_edge():
    """Test edge cases."""
    # n - p > 0 case with a zero-residual fit (sigma2_hat = 0 => se = 0).
    X = np.asarray([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]],
                   dtype=float)
    y = np.asarray([3.0, -1.0, 3.0, -1.0], dtype=float)
    beta = np.asarray([1.0, 2.0], dtype=float)
    result = esl_se_beta(X, y, beta)
    assert isinstance(result, dict)
    assert result["sigma2_hat"] == 0.0
    assert result["se"] == [0.0, 0.0]
    assert result["estimate"] == 0.0
    assert result["df_residual"] == 2
