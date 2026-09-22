"""Tests for b110.burkov_lm_ch1_dataset_bce."""

from morie.fn import _array_core as np

from morie.fn.b110 import burkov_lm_ch1_dataset_bce


def test_b110_basic():
    """Test basic functionality."""
    y_hat = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = burkov_lm_ch1_dataset_bce(y_hat, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_b110_edge():
    """Test edge cases."""
    y_hat = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    y = np.array([float(v) for v in np.random.default_rng(42).integers(0, 2, 100).tolist()])
    result = burkov_lm_ch1_dataset_bce(y_hat, y)
    assert isinstance(result, dict)
