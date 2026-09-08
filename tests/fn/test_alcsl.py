"""Tests for alcsl.alammar_cosine_similarity_loss."""

from morie.fn.alcsl import alammar_cosine_similarity_loss


def test_alcsl_basic():
    out = alammar_cosine_similarity_loss([[1.0, 0.0]], [[1.0, 0.0]], [1.0])
    assert out["estimate"] == 0.0


def test_alcsl_edge():
    import pytest
    with pytest.raises(ValueError, match="lie in"):
        alammar_cosine_similarity_loss([[1.0]], [[1.0]], [2.0])
