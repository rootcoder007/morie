"""Tests for eslr2.esl_r_squared."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.eslr2 import esl_r_squared


def test_eslr2_basic():
    """Test basic functionality with a design matrix and coefficient vector."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, (100, 5))
    y = rng.normal(0.0, 1.0, 100)
    beta = rng.normal(0.0, 1.0, 5)

    result = esl_r_squared(X, y, beta)

    assert isinstance(result, dict)
    assert "estimate" in result
    for key in ("rss", "tss", "n", "p", "method"):
        assert key in result

    assert result["n"] == 100
    assert result["p"] == 5

    # Independent computation of R^2 = 1 - RSS/TSS from the documented formula.
    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    residuals = y_arr - X_arr @ beta_arr
    rss_manual = float(np.sum(residuals ** 2))
    y_mean = float(np.mean(y_arr))
    tss_manual = float(np.sum((y_arr - y_mean) ** 2))
    expected_r2 = 1.0 - rss_manual / tss_manual

    assert np.isclose(result["estimate"], expected_r2)
    assert np.isclose(result["rss"], rss_manual)
    assert np.isclose(result["tss"], tss_manual)


def test_eslr2_edge():
    """Test edge cases: perfect fit gives R^2 == 1; constant y gives nan."""
    X = np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.asarray([1.0, 3.0, 5.0])

    # Perfect fit: beta = [1, 2] reproduces y exactly.
    perfect = esl_r_squared(X, y, np.asarray([1.0, 2.0]))
    assert isinstance(perfect, dict)
    assert "estimate" in perfect
    assert perfect["estimate"] == 1.0

    # Mean-only fit: beta = [mean(y), 0] -> R^2 == 0.
    mean_only = esl_r_squared(X, y, np.asarray([float(np.mean(y)), 0.0]))
    assert isinstance(mean_only, dict)
    assert mean_only["estimate"] == 0.0

    # Constant response: R^2 is undefined -> nan, not an error.
    y_const = np.asarray([2.0, 2.0, 2.0])
    const_res = esl_r_squared(X, y_const, np.asarray([2.0, 0.0]))
    assert isinstance(const_res, dict)
    assert np.isnan(const_res["estimate"])
    assert const_res["tss"] == 0.0
