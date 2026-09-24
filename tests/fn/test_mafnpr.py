"""Tests for mafnpr.ma_funnel_plot_data."""

from morie.fn import _array_core as np

from morie.fn.mafnpr import ma_funnel_plot_data


def test_mafnpr_basic():
    """Test basic functionality."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    se_i = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_funnel_plot_data(yi, se_i)
    assert isinstance(result, dict)
    assert "x_funnel" in result


def test_mafnpr_edge():
    """Test edge cases."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    se_i = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_funnel_plot_data(yi, se_i)
    assert isinstance(result, dict)
