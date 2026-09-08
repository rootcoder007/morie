"""Tests for albtm.alammar_bertopic_pipeline."""

from morie.fn.albtm import alammar_bertopic_pipeline


def test_albtm_basic():
    docs = [["cat", "dog", "cat"], ["dog", "cat"],
            ["stock", "bond", "stock"], ["bond", "stock", "bond"]]
    out = alammar_bertopic_pipeline(docs, [[0, 0], [0.1, 0.1],
                                           [5, 5], [5.1, 5.1]], 2)
    assert out["n_topics"] == 2


def test_albtm_edge():
    import pytest
    with pytest.raises(ValueError, match="one embedding"):
        alammar_bertopic_pipeline([["a"], ["b"], ["c"], ["d"]], [[0, 0]], 2)
