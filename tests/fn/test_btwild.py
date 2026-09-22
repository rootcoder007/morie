"""Tests for btwild.boot_wild_regression."""

from morie.fn import _array_core as np

from morie.fn.btwild import boot_wild_regression


def _to_2d_list(X):
    """Convert numpy array (or array-like) to list-of-lists, respecting shape."""
    return [list(row) for row in X]


def _to_1d_list(y):
    """Convert 1-D numpy array to flat list."""
    return [float(v) for v in y]


def test_btwild_basic():
    """Test basic functionality."""
    rng_X = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    X = _to_2d_list(rng_X.normal(0, 1, (100, 5)))
    y = _to_1d_list(rng_y.normal(0, 1, 100))
    B = 200
    result = boot_wild_regression(X, y, B)
    # result is a RichResult (dict-like); check key shape, not isinstance dict
    assert "estimate" in result
    assert "beta_hat" in result
    assert "beta_b" in result
    assert "se" in result
    assert "lo" in result
    assert "hi" in result
    assert "var_hc0" in result
    assert "v_mean" in result
    assert "v_var" in result
    assert "v_m3" in result
    assert "n" in result and "p" in result and "B" in result
    assert result["n"] == 100
    assert result["p"] == 5
    assert result["B"] == 200
    # beta_b must hold B replicate coefficient vectors, each of length p
    assert len(result["beta_b"]) == 200
    for row in result["beta_b"]:
        assert len(row) == 5
    # beta_hat is length p
    assert len(result["beta_hat"]) == 5
    assert len(result["se"]) == 5
    assert len(result["lo"]) == 5
    assert len(result["hi"]) == 5
    assert len(result["var_hc0"]) == 5


def test_btwild_edge():
    """Test edge cases."""
    rng_X = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    X = _to_2d_list(rng_X.normal(0, 1, (100, 5)))
    y = _to_1d_list(rng_y.normal(0, 1, 100))
    B = 50
    result = boot_wild_regression(X, y, B)
    assert "estimate" in result
    assert result["n"] == 100
    assert result["p"] == 5
    assert result["B"] == 50
    assert len(result["beta_b"]) == 50
