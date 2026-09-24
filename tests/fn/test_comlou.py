"""Tests for comlou.louvain_communities."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.comlou import louvain_communities


def test_comlou_basic():
    """Test basic functionality."""
    G = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    result = louvain_communities(G, 1.0, 5)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert "z" in result
    assert len(result["z"]) == 6


def test_comlou_edge():
    """Test edge cases."""
    G = [[0, 1, 0], [1, 0, 1]]
    with pytest.raises(ValueError):
        louvain_communities(G, 1.0)
