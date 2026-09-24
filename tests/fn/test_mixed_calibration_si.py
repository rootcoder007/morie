"""Tests for mixed_calibration_si.mixed_calibration_si."""

from morie.fn import _array_core as np

from morie.fn.mixed_calibration_si import (
    mixed_calibration_si,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e40_basic():
    """Test basic functionality."""
    z_sample = 0.5
    b_si = 0.5
    m_all_mean = 0.5
    m_sample_mean = 0.5
    result = mixed_calibration_si(z_sample, b_si, m_all_mean, m_sample_mean)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e40_edge():
    """Test edge cases."""
    z_sample = 0.5
    b_si = 0.5
    m_all_mean = 0.5
    m_sample_mean = 0.5
    result = mixed_calibration_si(z_sample, b_si, m_all_mean, m_sample_mean)
    assert isinstance(result, dict)
