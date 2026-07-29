"""Tests for bktfid.burkov_tf_idf."""

from morie.fn.bktfid import burkov_tf_idf


def test_bktfid_basic():
    corpus = [["a", "b"], ["b", "c"], ["b"]]
    assert burkov_tf_idf("b", ["a", "b"], corpus)["estimate"] == 0.0


def test_bktfid_edge():
    import pytest
    with pytest.raises(ValueError, match="no corpus document"):
        burkov_tf_idf("z", ["z"], [["a"]])
