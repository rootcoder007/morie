"""Tests for alndcg.alammar_ndcg_at_k."""

from morie.fn.alndcg import alammar_ndcg_at_k


def test_alndcg_basic():
    assert alammar_ndcg_at_k([3, 2, 1], 3)["estimate"] == 1.0


def test_alndcg_edge():
    import pytest
    with pytest.raises(ValueError, match="undefined"):
        alammar_ndcg_at_k([0, 0], 2)
