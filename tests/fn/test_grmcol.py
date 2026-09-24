"""Tests for grmcol.geron_gan_mode_collapse_metric."""

from morie.fn import _array_core as np

from morie.fn.grmcol import geron_gan_mode_collapse_metric


def test_grmcol_basic():
    """Test basic functionality."""
    samples = [[0.05], [-0.05], [10.1], [5.0]]
    true_modes = [[0.0], [10.0], [20.0]]
    result = geron_gan_mode_collapse_metric(samples, true_modes)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grmcol_edge():
    """Test edge cases."""
    samples = [[0.05], [-0.05], [10.1], [5.0]]
    true_modes = [[0.0], [10.0], [20.0]]
    result = geron_gan_mode_collapse_metric(samples, true_modes)
    assert isinstance(result, dict)
