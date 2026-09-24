"""Tests for survbri.brier_score."""

from morie.fn import _array_core as np

from morie.fn.survbri import brier_score


def test_survbri_basic():
    """Test basic functionality."""
    time = 0.5
    event = 1
    predicted_S = 0.5
    t_grid = 0.5
    result = brier_score(time, event, predicted_S, t_grid)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_survbri_edge():
    """Test edge cases."""
    time = 0.5
    event = 1
    predicted_S = 0.5
    t_grid = 0.5
    result = brier_score(time, event, predicted_S, t_grid)
    assert isinstance(result, dict)
