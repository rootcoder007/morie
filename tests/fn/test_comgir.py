"""Girvan-Newman on two triangles joined by one edge: the bridge goes first."""

from morie.fn.comgir import girvan_newman


def test_comgir_splits_at_the_bridge():
    A = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    r = girvan_newman(A)
    assert r["removed"][0] == (2, 3)
    assert r["labels"] == [0, 0, 0, 1, 1, 1]
    assert abs(r["modularity"] - 5 / 14) < 1e-12
