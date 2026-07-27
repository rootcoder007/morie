"""Tests for sgtnbe.sgt_nonbacktracking_matrix."""

import numpy as np
import pytest

from morie.fn.sgtnbe import sgt_nonbacktracking_matrix


def test_sgtnbe_cycle_graph_gives_a_permutation_matrix():
    """On a cycle every directed edge has exactly one non-backtracking
    continuation, so B is a permutation matrix: every row and column
    sums to 1. A hand-checkable structural fact."""
    n = 5
    edges = [(i, (i + 1) % n) for i in range(n)]
    r = sgt_nonbacktracking_matrix(edges)
    B = np.asarray(r["B"])
    assert B.shape == (2 * n, 2 * n)
    np.testing.assert_array_equal(B.sum(axis=1), np.ones(2 * n))
    np.testing.assert_array_equal(B.sum(axis=0), np.ones(2 * n))


def test_sgtnbe_never_walks_straight_back():
    """B[(u,v), (v,u)] = 0 for every edge -- the defining constraint."""
    edges = [(0, 1), (1, 2), (2, 0), (1, 3)]
    r = sgt_nonbacktracking_matrix(edges)
    B = np.asarray(r["B"])
    pos = {e: i for i, e in enumerate(r["directed_edges"])}
    for (u, v), i in pos.items():
        assert B[i, pos[(v, u)]] == 0


def test_sgtnbe_path_graph_endpoints_dead_end():
    """On a path, the directed edge arriving at an endpoint has nowhere to
    continue: its row of B is all zero."""
    r = sgt_nonbacktracking_matrix([(0, 1), (1, 2)])
    B = np.asarray(r["B"])
    pos = {e: i for i, e in enumerate(r["directed_edges"])}
    assert B[pos[(1, 2)]].sum() == 0  # arrived at endpoint 2
    assert B[pos[(1, 0)]].sum() == 0  # arrived at endpoint 0
    assert B[pos[(0, 1)]].sum() == 1  # must continue to (1, 2)


def test_sgtnbe_row_sums_are_out_degree_minus_one_off_endpoints():
    """For an edge arriving at a vertex of degree d, there are d - 1
    continuations."""
    star = [(0, 1), (0, 2), (0, 3)]  # centre 0 has degree 3
    r = sgt_nonbacktracking_matrix(star)
    B = np.asarray(r["B"])
    pos = {e: i for i, e in enumerate(r["directed_edges"])}
    assert B[pos[(1, 0)]].sum() == 2  # continue to (0,2) or (0,3)
