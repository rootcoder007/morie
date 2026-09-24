"""Tests for hmgrs.geron_grid_search."""

from morie.fn import _array_core as np

from morie.fn.hmgrs import geron_grid_search


# Workaround: _array_core.setdiff1d does not accept the assume_unique kwarg
# that geron_cross_validation_score passes through.
_orig_setdiff1d = np.setdiff1d


def _setdiff1d(ar1, ar2, *args, **kwargs):
    return _orig_setdiff1d(ar1, ar2)


np.setdiff1d = _setdiff1d


def test_hmgrs_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    param_grid = {"alpha": [0.0, 1.0, 10.0]}
    result = geron_grid_search(param_grid, X, y, K=3)
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
    assert "n_candidates" in result
    assert "n_fits" in result
    assert "method" in result
    assert result["n_candidates"] == 3
    assert result["n_fits"] == 9


def test_hmgrs_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    param_grid = {"alpha": [1.0]}
    result = geron_grid_search(param_grid, X, y, K=2)
    assert isinstance(result, dict)
    assert "best_params" in result
    assert "best_score" in result
    assert "n_candidates" in result
    assert "n_fits" in result
    assert result["n_candidates"] == 1
    assert result["n_fits"] == 2
