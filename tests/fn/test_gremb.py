"""Tests for gremb.geron_embedding_lookup."""

import pytest

from morie.fn import _array_core as np
from morie.fn.gremb import geron_embedding_lookup


def test_gremb_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    V, d = 50, 8
    E = rng.normal(0, 1, (V, d))
    n = 20
    ids = rng.integers(0, V, n)
    result = geron_embedding_lookup(ids, E)
    assert isinstance(result, dict)
    assert "embeddings" in result
    assert result["vocab_size"] == V
    assert result["dim"] == d
    assert result["n"] == n
    assert len(result["embeddings"]) == n
    assert all(len(row) == d for row in result["embeddings"])


def test_gremb_edge():
    """Test edge cases - repeated ids and out-of-range error."""
    rng = np.random.default_rng(42)
    V, d = 5, 3
    E = rng.normal(0, 1, (V, d))
    # Repeated ids
    ids = [1, 1, 1]
    result = geron_embedding_lookup(ids, E)
    assert isinstance(result, dict)
    assert result["n_unique"] == 1
    assert result["n"] == 3
    # Out-of-range error
    with pytest.raises(ValueError):
        geron_embedding_lookup([V], E)
