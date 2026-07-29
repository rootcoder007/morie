"""Tests for almnrl.alammar_multiple_negatives_ranking."""

from morie.fn.almnrl import alammar_multiple_negatives_ranking


def test_almnrl_basic():
    A = [[1.0, 0.0], [0.0, 1.0]]
    out = alammar_multiple_negatives_ranking(A, A, tau=1.0)
    assert out["estimate"] > 0


def test_almnrl_edge():
    import pytest
    with pytest.raises(ValueError, match="at least 2"):
        alammar_multiple_negatives_ranking([[1.0]], [[1.0]])
