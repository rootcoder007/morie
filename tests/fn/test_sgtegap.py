"""Tests for sgtegap.sgt_eigengap_heuristic."""

from morie.fn import _array_core as np

from morie.fn.sgtegap import sgt_eigengap_heuristic


def test_sgtegap_basic():
    """Test basic functionality."""
    values = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = sgt_eigengap_heuristic(values)
    assert isinstance(result, dict)
    assert "k" in result


def test_sgtegap_edge():
    """Test edge cases."""
    values = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = sgt_eigengap_heuristic(values)
    assert isinstance(result, dict)
