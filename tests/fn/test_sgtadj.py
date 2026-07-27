"""Tests for sgtadj.sgt_adjacency_matrix."""

import numpy as np
import pytest

from morie.fn.sgtadj import sgt_adjacency_matrix


def test_sgtadj_builds_the_path_graph():
    r = sgt_adjacency_matrix([("A", "B"), ("B", "C")])
    A = np.asarray(r["A"])
    want = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    np.testing.assert_array_equal(A, want)
    assert r["nodes"] == {"A": 0, "B": 1, "C": 2}
    np.testing.assert_array_equal(r["degree"], [1, 2, 1])
    assert r["m"] == 2


def test_sgtadj_directed_keeps_edges_one_way():
    r = sgt_adjacency_matrix([(0, 1), (1, 2)], n=3, directed=True)
    A = np.asarray(r["A"])
    assert A[0, 1] == 1 and A[1, 0] == 0
    assert A[1, 2] == 1 and A[2, 1] == 0


def test_sgtadj_undirected_matrix_is_symmetric_and_deduplicated():
    r = sgt_adjacency_matrix([(0, 1), (1, 0), (0, 1)], n=4)
    A = np.asarray(r["A"])
    np.testing.assert_array_equal(A, A.T)
    assert A.sum() == 2  # one undirected edge, both directions
    assert A.shape == (4, 4)  # isolated nodes preserved via n


def test_sgtadj_rejects_bad_edges_and_labels():
    with pytest.raises(ValueError, match="pair"):
        sgt_adjacency_matrix([(0, 1, 2)])
    with pytest.raises(ValueError, match="lie in"):
        sgt_adjacency_matrix([(0, 5)], n=3)
