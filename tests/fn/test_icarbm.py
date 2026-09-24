"""Tests for icarbm.icar_prior."""

import pytest

from morie.fn import _array_core as np

from morie.fn.icarbm import icar_prior


def test_icarbm_basic():
    """Test basic functionality."""
    # Symmetric adjacency matrix with zero diagonal (triangle graph)
    adjacency = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    tau = 0.5
    result = icar_prior(adjacency, tau)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "precision" in result
    assert "conditional_mean" in result
    assert "conditional_var" in result
    assert "pairwise_quadratic" in result
    assert "log_density_kernel" in result
    assert "smallest_eigenvalue" in result
    assert "mean_u" in result
    assert "n" in result
    assert result["n"] == 3
    # Precision matrix should be 3x3
    assert len(result["precision"]) == 3
    assert all(len(row) == 3 for row in result["precision"])
    # Conditional variance vector has one entry per unit
    assert len(result["conditional_var"]) == 3
    assert len(result["conditional_mean"]) == 3


def test_icarbm_edge():
    """Test edge cases: non-symmetric adjacency must be rejected."""
    bad_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    with pytest.raises(ValueError):
        icar_prior(bad_adj, 0.5)
