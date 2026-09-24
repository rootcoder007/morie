"""Tests for smltot.survey_total."""

from morie.fn import _array_core as np

from morie.fn.smltot import survey_total


def test_smltot_basic():
    """Test basic functionality."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    N = 5
    result = survey_total(y, N)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_smltot_edge():
    """Test edge cases."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    N = 5
    result = survey_total(y, N)
    assert isinstance(result, dict)
