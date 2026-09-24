"""Tests for otmot.ot_multimarginal_iter."""

from morie.fn import _array_core as np

from morie.fn.otmot import ot_multimarginal_iter


def test_otmot_basic():
    """Test basic functionality."""
    margins = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    C_tensor = 0.5
    epsilon = 0.5
    result = ot_multimarginal_iter(margins, C_tensor, epsilon)
    assert isinstance(result, dict)
    assert "T" in result


def test_otmot_edge():
    """Test edge cases."""
    margins = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    C_tensor = 0.5
    epsilon = 0.5
    result = ot_multimarginal_iter(margins, C_tensor, epsilon)
    assert isinstance(result, dict)
