"""Tests for dfbetb.dfbetas."""

from morie.fn import _array_core as np

from morie.fn.dfbetb import dfbetas


def test_dfbetb_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    y = rng_y.normal(0, 1, 100)
    X = rng_X.normal(0, 1, (100, 5))
    # The shim's mat() does list(float(row)) on each row, which fails for
    # multi-element arrays. Convert each row/column to plain Python lists
    # before passing in, so the function sees list-of-lists of floats.
    y_list = [float(v) for v in y]
    X_list = [[float(v) for v in row] for row in X]
    result = dfbetas(y_list, X_list)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "dfbetas" in result
    assert "threshold" in result
    assert "n_influential" in result
    assert "n" in result
    assert "p" in result
    assert result["n"] == 100
    assert result["p"] == 6  # intercept + 5 predictors
    # dfbetas must be n x p
    dfb = result["dfbetas"]
    assert len(dfb) == 100
    assert all(len(row) == 6 for row in dfb)
    # estimate equals the max absolute value across the matrix
    expected_estimate = max(abs(v) for row in dfb for v in row)
    assert result["estimate"] == expected_estimate
    # threshold = 2 / sqrt(n)
    n = 100
    expected_threshold = 2.0 / (n ** 0.5)
    assert result["threshold"] == expected_threshold
    # n_influential counts rows with any |DFBETAS| > threshold
    thr = expected_threshold
    expected_n_infl = sum(
        1 for row in dfb if any(abs(v) > thr for v in row)
    )
    assert result["n_influential"] == expected_n_infl


def test_dfbetb_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    y = rng_y.normal(0, 1, 100)
    X = rng_X.normal(0, 1, (100, 5))
    y_list = [float(v) for v in y]
    X_list = [[float(v) for v in row] for row in X]
    result = dfbetas(y_list, X_list)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "dfbetas" in result
