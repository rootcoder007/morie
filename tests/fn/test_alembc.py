"""Tests for alembc.alammar_embedding_classifier."""

from morie.fn.alembc import alammar_embedding_classifier


def test_alembc_basic():
    out = alammar_embedding_classifier([[0, 0], [0.1, 0], [5, 5], [5.1, 5]],
                                       [0, 0, 1, 1])
    assert out["train_accuracy"] == 1.0


def test_alembc_edge():
    import pytest
    with pytest.raises(ValueError, match="at least 2 classes"):
        alammar_embedding_classifier([[0, 0], [1, 1]], [0, 0])
