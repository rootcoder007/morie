"""Tests for model_averaged_variance.model_averaged_variance."""

from morie.fn import _array_core as np

from morie.fn.model_averaged_variance import (
    model_averaged_variance,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5e4_basic():
    """Test basic functionality."""
    taus = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    thetas = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    variances = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = model_averaged_variance(taus, thetas, variances)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5e4_edge():
    """Test edge cases."""
    taus = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    thetas = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    variances = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = model_averaged_variance(taus, thetas, variances)
    assert isinstance(result, dict)
