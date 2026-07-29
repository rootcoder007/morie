"""Tests for alumap.alammar_umap_projection."""

from morie.fn.alumap import alammar_umap_projection


def test_alumap_basic():
    X = [[0, 0], [0.2, 0], [0, 0.2], [8, 8], [8.2, 8], [8, 8.2]]
    out = alammar_umap_projection(X, n_neighbors=2, n_steps=50)
    assert out["objective_decreased"] is True


def test_alumap_edge():
    import pytest
    with pytest.raises(ValueError, match="n_neighbors"):
        alammar_umap_projection([[0, 0], [1, 1]], n_neighbors=5)
