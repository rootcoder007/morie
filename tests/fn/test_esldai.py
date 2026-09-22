"""Tests for esldai.esl_dirichlet_proc."""

from morie.fn import _array_core as np

from morie.fn.esldai import esl_dirichlet_proc


def test_esldai_basic():
    """Test basic functionality."""
    result = esl_dirichlet_proc()
    assert isinstance(result, dict)
    assert "weights" in result
def test_esldai_edge():
    """Test edge cases."""
    result = esl_dirichlet_proc()
    assert isinstance(result, dict)
