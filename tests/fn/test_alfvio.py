"""Tests for alfvio.alphafold_violation."""

from morie.fn import _array_core as np

from morie.fn.alfvio import alphafold_violation


def test_alfvio_basic():
    """Test basic functionality."""
    result = alphafold_violation()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_alfvio_edge():
    """Test edge cases."""
    result = alphafold_violation()
    assert isinstance(result, dict)
