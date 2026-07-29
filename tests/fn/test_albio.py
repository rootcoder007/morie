"""Tests for albio.alammar_bio_tagging."""

from morie.fn.albio import alammar_bio_tagging


def test_albio_basic():
    out = alammar_bio_tagging(["a", "b", "c"], [(0, 2, "PER")])
    assert out["tags"] == ["B-PER", "I-PER", "O"]


def test_albio_edge():
    import pytest
    with pytest.raises(ValueError, match="overlap"):
        alammar_bio_tagging(["a", "b"], [(0, 2, "X"), (1, 2, "Y")])
