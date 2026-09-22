"""Tests for dffit.dffits."""

from morie.fn import _array_core as np

from morie.fn.dffit import dffits


def _to_list(a):
    """Convert numpy-like array to nested Python lists."""
    return [list(row) if hasattr(row, "__iter__") else row for row in a]


def test_dffit_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    y = rng_y.normal(0, 1, 100)
    X = rng_X.normal(0, 1, (100, 5))

    y_list = _to_list(y)
    X_list = _to_list(X)

    result = dffits(y_list, X_list)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    # Documented return keys
    assert "dffits" in result
    assert "threshold" in result
    assert "flagged" in result
    assert "n_influential" in result
    assert "n" in result
    assert "p" in result


def test_dffit_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    y = rng_y.normal(0, 1, 100)
    X = rng_X.normal(0, 1, (100, 5))

    y_list = _to_list(y)
    X_list = _to_list(X)

    result = dffits(y_list, X_list)
    assert isinstance(result, dict)
    assert result["n"] == 100
    assert result["p"] == 6  # 5 predictors + intercept
    assert len(result["dffits"]) == 100
    assert len(result["flagged"]) == 100
    assert result["n_influential"] == sum(result["flagged"])
