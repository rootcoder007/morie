"""Tests for pmedex.proportion_te_explained."""

from morie.fn import _array_core as np

from morie.fn.pmedex import proportion_te_explained


def test_pmedex_basic():
    """Test basic functionality."""
    nie = 0.1
    te = 0.1
    result = proportion_te_explained(nie, te)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_pmedex_edge():
    """Test edge cases."""
    nie = 0.1
    te = 0.1
    result = proportion_te_explained(nie, te)
    assert isinstance(result, dict)
