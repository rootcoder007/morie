"""Tests for grbvd.geron_bias_variance_decomposition."""

from morie.fn import _array_core as np

from morie.fn.grbvd import geron_bias_variance_decomposition


def test_grbvd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_samples = 50
    n_models = 10
    # predictions: 2-D matrix of predictions from n_models, each predicting n_samples
    predictions = rng.normal(0, 1, (n_models, n_samples))
    y_true = rng.normal(0, 1, n_samples)
    result = geron_bias_variance_decomposition(predictions, y_true)
    assert isinstance(result, dict)
    # The decomposition should expose at least one bias-variance-noise component
    assert any(
        k in result
        for k in ("bias", "variance", "noise", "expected_loss",
                  "irreducible_error", "total_error", "avg_bias", "avg_variance")
    )


def test_grbvd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n_samples = 20
    n_models = 5
    predictions = rng.normal(0, 1, (n_models, n_samples))
    y_true = rng.normal(0, 1, n_samples)
    result = geron_bias_variance_decomposition(predictions, y_true)
    assert isinstance(result, dict)
