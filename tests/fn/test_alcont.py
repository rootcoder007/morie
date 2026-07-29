"""Tests for alcont.alammar_continued_pretraining_mlm."""

from morie.fn.alcont import alammar_continued_pretraining_mlm


def test_alcont_basic():
    out = alammar_continued_pretraining_mlm(["d"],
        lambda docs, s: 1.0 / (s + 1), 3)
    assert out["mlm_improved"] is True


def test_alcont_edge():
    import pytest
    with pytest.raises(ValueError, match="empty"):
        alammar_continued_pretraining_mlm([], lambda d, s: 1.0, 3)
