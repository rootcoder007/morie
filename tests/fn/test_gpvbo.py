"""Tests for gpvbo.gp_variational_bayes_opt."""

from morie.fn import _array_core as np

from morie.fn.gpvbo import gp_variational_bayes_opt


def test_gpvbo_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_g = np.random.default_rng(44)
    # X must be a list of 1-D point vectors; use 1-D inputs so dim matches.
    X = [[float(rng_x.normal(0, 1))] for _ in range(10)]
    y = [float(rng_y.normal(0, 1)) for _ in range(10)]
    X_grid = [[float(rng_g.normal(0, 1))] for _ in range(20)]
    result = gp_variational_bayes_opt(X, y, X_grid)
    assert isinstance(result, dict)
    # Documented payload keys (per function source).
    for key in ("estimate", "acquisition", "mean", "sd",
                "next_index", "next_point", "f_min", "n", "method"):
        assert key in result
    assert isinstance(result["acquisition"], list)
    assert isinstance(result["mean"], list)
    assert isinstance(result["sd"], list)
    assert isinstance(result["next_point"], list)
    assert result["n"] == len(X)
    assert len(result["acquisition"]) == len(X_grid)
    assert len(result["mean"]) == len(X_grid)
    assert len(result["sd"]) == len(X_grid)
    # `estimate` equals the maximum of the acquisition list.
    assert result["estimate"] == max(result["acquisition"])
    # `next_index` is 1-based position of the argmax in X_grid.
    best = 0
    for j in range(len(result["acquisition"])):
        if result["acquisition"][j] > result["acquisition"][best]:
            best = j
    assert result["next_index"] == best + 1
    assert result["next_point"] == X_grid[best]
    # f_min is min of y.
    assert result["f_min"] == min(y)
    # All acquisition values are non-negative.
    assert all(ei >= 0.0 for ei in result["acquisition"])
    # sd values are non-negative.
    assert all(s >= 0.0 for s in result["sd"])


def test_gpvbo_edge():
    """Test edge cases (matching the basic shape contract)."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_g = np.random.default_rng(44)
    X = [[float(rng_x.normal(0, 1))] for _ in range(5)]
    y = [float(rng_y.normal(0, 1)) for _ in range(5)]
    X_grid = [[float(rng_g.normal(0, 1))] for _ in range(8)]
    result = gp_variational_bayes_opt(X, y, X_grid)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "next_point" in result
    assert "f_min" in result
    assert result["n"] == 5
    assert len(result["acquisition"]) == 8
