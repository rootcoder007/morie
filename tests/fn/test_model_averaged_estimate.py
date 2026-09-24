"""Tests for model_averaged_estimate.model_averaged_estimate."""

from morie.fn import _array_core as np

from morie.fn.model_averaged_estimate import (
    model_averaged_estimate,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5e3_basic():
    """Test basic functionality."""
    taus = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    thetas = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = model_averaged_estimate(taus, thetas)
    assert isinstance(result, dict)
    assert "value" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo5e3_edge():
    """Test edge cases."""
    taus = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    thetas = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = model_averaged_estimate(taus, thetas)
    assert isinstance(result, dict)
