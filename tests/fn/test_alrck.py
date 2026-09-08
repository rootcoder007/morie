"""Tests for alrck.alammar_recall_at_k."""

from morie.fn.alrck import alammar_recall_at_k


def test_alrck_basic():
    assert alammar_recall_at_k([1, 2, 3, 4], [2, 9], 3)["estimate"] == 0.5


def test_alrck_edge():
    import pytest
    with pytest.raises(ValueError, match="empty"):
        alammar_recall_at_k([1], [], 1)
