"""Tests for agrsc."""

from morie.fn import _array_core as np
import pytest

from morie.fn.agrmt import agreement_score
from morie.fn.agrsc import agreement_score_matrix


def test_agrsc_basic():
    V = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, np.nan]])
    out = agreement_score_matrix(V)
    assert out["matrix"] == pytest.approx(agreement_score(V).value["agreement_matrix"])
    assert out["matrix"][0, 2] == pytest.approx(1.0)


def test_agrsc_edge():
    with pytest.raises(ValueError):
        agreement_score_matrix(np.ones(4))  # 1-D input
