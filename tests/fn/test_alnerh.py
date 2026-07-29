"""Tests for alnerh.alammar_ner_token_head."""

from morie.fn.alnerh import alammar_ner_token_head


def test_alnerh_basic():
    out = alammar_ner_token_head([[1.0, 0.0]], [[2.0, 0.0], [0.0, 2.0]],
                                 [0.0, 0.0], [0])
    assert out["predicted_tags"] == [0]


def test_alnerh_edge():
    import pytest
    with pytest.raises(ValueError, match="out of range"):
        alammar_ner_token_head([[1.0, 0.0]], [[2.0, 0.0]], [0.0], [5])
