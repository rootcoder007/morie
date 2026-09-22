"""Tests for eslrss.esl_residual_sum_squares."""

from morie.fn import _array_core as np

from morie.fn.eslrss import esl_residual_sum_squares


def test_eslrss_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    beta = np.random.default_rng(44).normal(0, 1, 5)
    result = esl_residual_sum_squares(X, y, beta)
    assert isinstance(result, dict)
    assert "estimate" in result
    # Compute the expected RSS from the documented formula:
    # RSS(beta) = sum_i (y_i - x_i' beta)^2
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    resid_expected = y_arr - X_arr @ beta_arr
    rss_expected = float(resid_expected @ resid_expected)
    assert result["estimate"] == rss_expected
    assert result["n"] == 100
    assert result["p"] == 5
    assert result["mean_squared_error"] == rss_expected / 100
    assert result["method"].startswith("RSS")


def test_eslrss_edge():
    """Test edge cases."""
    # Exact-fit case from the docstring example:
    # X = [[1, 0], [1, 1], [1, 2]], y = [1, 3, 5], beta = [1, 2]
    # gives predictions [1, 3, 5] and RSS = 0.
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 3.0, 5.0])
    beta = np.array([1.0, 2.0])
    result = esl_residual_sum_squares(X, y, beta)
    assert isinstance(result, dict)
    assert result["estimate"] == 0.0
    assert result["n"] == 3
    assert result["p"] == 2
    # Unit shift of the intercept costs n, also from the docstring:
    # X, y as above with beta = [0, 2] -> predictions [0, 2, 4],
    # residuals [1, 1, 1], RSS = 3.
    beta_shift = np.array([0.0, 2.0])
    result_shift = esl_residual_sum_squares(X, y, beta_shift)
    resid_expected_shift = y - np.asarray(X, dtype=float) @ beta_shift
    rss_expected_shift = float(resid_expected_shift @ resid_expected_shift)
    assert result_shift["estimate"] == rss_expected_shift
    assert result_shift["estimate"] == 3.0
    assert result_shift["mean_squared_error"] == rss_expected_shift / 3
