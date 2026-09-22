"""Tests for eslzst.esl_z_score."""

from morie.fn import _array_core as np

from morie.fn.eslzst import esl_z_score


def test_eslzst_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (20, 5))
    # Set first column to 1 (intercept) so X[:,0] = intercept
    X[:, 0] = 1.0
    # Generate y from a known linear model so we can compute expected beta
    true_beta = np.array([1.0, 0.8, -0.5, 0.3, 0.0])
    y = X @ true_beta + rng.normal(0, 1, 20)
    # Fit OLS via lstsq
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    result = esl_z_score(X, y, beta)
    assert isinstance(result, dict)
    # Documented keys
    for key in ("estimate", "z", "se", "p_normal", "p_t", "df", "n", "p", "method"):
        assert key in result, f"missing key {key}"
    # Shapes
    assert len(result["z"]) == 5
    assert len(result["se"]) == 5
    assert len(result["p_normal"]) == 5
    assert len(result["p_t"]) == 5
    # estimate equals first z
    assert result["estimate"] == result["z"][0]
    # n, p, df as documented
    assert result["n"] == 20
    assert result["p"] == 5
    assert result["df"] == 20 - 5
    # z = beta / se element-wise (compute se from z and beta)
    se_from_z = beta / np.array(result["z"])
    assert np.allclose(se_from_z, np.array(result["se"]))


def test_eslzst_edge():
    """Test edge cases: perfect fit gives inf z and zero p-values."""
    # Perfectly linear data: residuals vanish, so one SE -> 0 and z -> inf
    X = np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]])
    y = np.array([3.0, -1.0, 3.0, -1.0])
    beta = np.array([1.0, 2.0])
    result = esl_z_score(X, y, beta)
    assert isinstance(result, dict)
    # The non-intercept coefficient has perfect fit -> inf z
    assert result["z"][1] == np.inf
    # Its tail probability should be 0 (documented behaviour)
    assert result["p_t"][1] == 0.0
    assert result["p_normal"][1] == 0.0
    # df = n - p = 4 - 2 = 2
    assert result["df"] == 2
    assert result["n"] == 4
    assert result["p"] == 2
