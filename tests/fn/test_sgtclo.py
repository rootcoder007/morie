"""Tests for sgtclo.sgt_closeness_centrality."""

from collections import deque

import pytest

from morie.fn.sgtclo import sgt_closeness_centrality

A = [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 0], [1, 0, 1, 0, 0]]


def _bfs(s):
    d = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in range(5):
            if A[u][v] and v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


def test_sgtclo_basic():
    """C(v) = (n - 1) / sum_u d(v, u) with BFS distances, recomputed."""
    r = sgt_closeness_centrality(A)
    exp = [4 / sum(_bfs(v).values()) for v in range(5)]
    assert r["closeness"] == pytest.approx(exp, rel=1e-15)
    assert r["argmax"] == exp.index(max(exp))


def test_sgtclo_edge():
    """A disconnected graph is refused."""
    with pytest.raises(ValueError):
        sgt_closeness_centrality([[0, 1, 0], [1, 0, 0], [0, 0, 0]])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.sgtclo as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
