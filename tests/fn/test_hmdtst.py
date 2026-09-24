"""Tests for hmdtst.geron_tree_sensitivity_scale."""

from morie.fn import _array_core as np

from morie.fn.hmdtst import geron_tree_sensitivity_scale


def test_hmdtst_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    X = rng_x.normal(0, 1, (40, 3))
    y = rng_y.normal(0, 1, 40)
    result = geron_tree_sensitivity_scale(X, y, criterion="mse")
    assert isinstance(result, dict)
    expected_keys = {
        "predictions_match", "thresholds", "scaled_thresholds",
        "expected_thresholds", "thresholds_match", "knn_predictions",
        "knn_scaled_predictions", "knn_match", "estimate", "n", "method"
    }
    assert expected_keys.issubset(set(result.keys()))


def test_hmdtst_edge():
    """Test edge cases."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    X = rng_x.normal(0, 1, (40, 3))
    y = rng_y.normal(0, 1, 40)
    result = geron_tree_sensitivity_scale(X, y, a=0.001, b=0.0, feature=1, criterion="mse")
    assert isinstance(result, dict)
    assert "predictions_match" in result
    assert "knn_match" in result
