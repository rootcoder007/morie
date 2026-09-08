"""Tests for alfrz.alammar_layer_freezing."""

from morie.fn.alfrz import alammar_layer_freezing


def test_alfrz_basic():
    out = alammar_layer_freezing(3)
    assert out["masks"][0] == [False, False, True]
    assert out["trainable_per_stage"] == [1, 2, 3]


def test_alfrz_edge():
    import pytest
    with pytest.raises(ValueError, match="n_stages"):
        alammar_layer_freezing(3, 5)
