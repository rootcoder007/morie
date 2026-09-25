"""Tests for sgtdeg.sgt_degree_matrix."""

import pytest

from morie.fn.sgtdeg import sgt_degree_matrix

A = [[0, 2.0, 0.5], [2.0, 0, 1.0], [0.5, 1.0, 0]]


def test_sgtdeg_basic():
    """d_v = sum_u A_uv, D = diag(d), vol = sum d."""
    r = sgt_degree_matrix(A)
    assert list(r["degrees"]) == [2.5, 3.0, 1.5]
    assert [list(row) for row in r["D"]] == [[2.5, 0, 0], [0, 3.0, 0], [0, 0, 1.5]]
    assert r["volume"] == 7.0


def test_sgtdeg_edge():
    """Negative weights are refused."""
    with pytest.raises(ValueError):
        sgt_degree_matrix([[0, -1.0], [-1.0, 0]])
