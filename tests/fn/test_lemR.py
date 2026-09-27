"""Tests for lemR.leiden_grph."""

from morie.fn.lemR import leiden_grph


def test_lemR_basic():
    """Test basic functionality."""
    A = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    result = leiden_grph(A)
    assert isinstance(result, dict)
    assert result["labels"] == [0, 0, 0, 1, 1, 1]
    assert result["connected"] and result["n_communities"] == 2
    assert abs(result["estimate"] - 5 / 14) < 1e-12  # two triangles joined by one edge: 2(6/14) - 2(7/14)^2


def test_lemR_edge():
    """Test edge cases."""
    A = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    result = leiden_grph(A)
    assert isinstance(result, dict)
