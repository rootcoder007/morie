"""Tests for albow.alammar_bag_of_words."""

from morie.fn.albow import alammar_bag_of_words


def test_albow_basic():
    out = alammar_bag_of_words(["a", "b", "a", "z"], ["a", "b", "c"])
    assert out["bow_vector"] == [2, 1, 0]
    assert out["oov_count"] == 1


def test_albow_edge():
    import pytest
    with pytest.raises(ValueError, match="duplicates"):
        alammar_bag_of_words(["a"], ["a", "a"])
