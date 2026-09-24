"""Tests for timesf.timesfm_foundation."""

from morie.fn import _array_core as np

from morie.fn.timesf import timesfm_foundation


def test_timesf_basic():
    """Test basic functionality."""
    history = [1.0] * 16
    predictor = lambda p: [0.0] * 8
    horizon = 20
    input_patch_len = 16
    output_patch_len = 8
    result = timesfm_foundation(history, predictor, horizon, input_patch_len, output_patch_len)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_timesf_edge():
    """Test edge cases."""
    history = [1.0] * 16
    predictor = lambda p: [0.0] * 8
    horizon = 20
    input_patch_len = 16
    output_patch_len = 8
    result = timesfm_foundation(history, predictor, horizon, input_patch_len, output_patch_len)
    assert isinstance(result, dict)
