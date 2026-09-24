"""Tests for colbrt.colbert."""

import math

from morie.fn import _array_core as np

from morie.fn.colbrt import colbert


def test_colbrt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    nq, d = 5, 4
    n_docs = 3
    nd = 4
    query = rng.normal(0, 1, (nq, d))
    docs = [rng.normal(0, 1, (nd, d)) for _ in range(n_docs)]
    result = colbert(query, docs)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "scores" in result
    assert "ranking" in result
    assert "best" in result
    assert "max_sim" in result
    assert "nq" in result
    assert "n_docs" in result
    assert result["nq"] == nq
    assert result["n_docs"] == n_docs
    assert len(result["scores"]) == n_docs
    assert len(result["ranking"]) == n_docs
    assert all(math.isfinite(s) for s in result["scores"])
    assert math.isfinite(result["estimate"])


def test_colbrt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    nq, d = 2, 3
    query = rng.normal(0, 1, (nq, d))
    doc = rng.normal(0, 1, (2, d))
    result = colbert(query, [doc])
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "scores" in result
    assert "ranking" in result
    assert "best" in result
    assert "max_sim" in result
    assert "nq" in result
    assert "n_docs" in result
    assert result["nq"] == nq
    assert result["n_docs"] == 1
    assert len(result["scores"]) == 1
    assert len(result["ranking"]) == 1
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["scores"][0])
