"""Tests for alclsh.alammar_classification_head."""

from morie.fn.alclsh import alammar_classification_head


def test_alclsh_basic():
    out = alammar_classification_head([1.0, 0.0],
        [[2.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
    assert out["predicted_class"] == 0


def test_alclsh_edge():
    import pytest
    with pytest.raises(ValueError, match="columns"):
        alammar_classification_head([1.0], [[1.0, 2.0]], [0.0])
