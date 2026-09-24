"""Tests for mabaujat.ma_baujat_plot_data."""

from morie.fn import _array_core as np

from morie.fn.mabaujat import ma_baujat_plot_data


def test_mabaujat_basic():
    """Test basic functionality."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_baujat_plot_data(yi, vi)
    assert isinstance(result, dict)
    assert "x" in result


def test_mabaujat_edge():
    """Test edge cases."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_baujat_plot_data(yi, vi)
    assert isinstance(result, dict)
