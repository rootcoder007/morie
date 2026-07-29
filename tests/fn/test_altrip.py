"""Tests for altrip.alammar_sbert_triplet_loss."""

from morie.fn.altrip import alammar_sbert_triplet_loss


def test_altrip_basic():
    out = alammar_sbert_triplet_loss([[0.0]], [[0.0]], [[5.0]], margin=1.0)
    assert out["estimate"] == 0.0
    assert out["active"] == [False]


def test_altrip_edge():
    import pytest
    with pytest.raises(ValueError, match="non-negative"):
        alammar_sbert_triplet_loss([[0.0]], [[0.0]], [[1.0]], margin=-1)
