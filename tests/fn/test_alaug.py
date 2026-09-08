"""Tests for alaug.alammar_augmented_sbert."""

from morie.fn.alaug import alammar_augmented_sbert


def test_alaug_basic():
    out = alammar_augmented_sbert([("x", "x")], lambda a, b: 1.0)
    assert out["n_silver"] == 1


def test_alaug_edge():
    import pytest
    with pytest.raises(ValueError, match="callable"):
        alammar_augmented_sbert([("x", "y")], "not a function")
