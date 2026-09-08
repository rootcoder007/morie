"""Tests for alldat.alammar_lda_topic_distribution."""

from morie.fn.alldat import alammar_lda_topic_distribution


def test_alldat_basic():
    docs = [["cat", "dog"], ["stock", "bond"]]
    out = alammar_lda_topic_distribution(docs, 2, n_iter=20)
    assert abs(sum(out["theta"][0]) - 1.0) < 1e-9


def test_alldat_edge():
    import pytest
    with pytest.raises(ValueError, match="at least 2"):
        alammar_lda_topic_distribution([["a"]], 1)
