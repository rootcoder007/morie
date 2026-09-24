"""Tests for krfgrp.kronecker_graph."""

from morie.fn import _array_core as np
from morie.fn.krfgrp import kronecker_graph


def test_krfgrp_basic():
    """Test basic functionality."""
    seed = np.array([[0.9, 0.5], [0.5, 0.3]])
    k = 5
    result = kronecker_graph(seed, k)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "expected_edges" in result
    assert "expected_self_loops" in result


def test_krfgrp_edge():
    """Test edge cases."""
    seed = np.array([[0.8, 0.6], [0.6, 0.2]])
    k = 1
    result = kronecker_graph(seed, k)
    assert isinstance(result, dict)
    assert "n_nodes" in result
    assert "mean_degree" in result
    assert "p_min" in result
