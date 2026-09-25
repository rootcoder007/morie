"""Tests for sgtsbms.sgt_sbm_spectral_estimate."""

import math

from morie.fn.sgtsbms import sgt_sbm_spectral_estimate


def _two_cliques(m):
    """Two disjoint cliques of size m: a planted partition with no noise."""
    n = 2 * m
    A = [[0.0] * n for _ in range(n)]
    for block in (range(m), range(m, n)):
        for i in block:
            for j in block:
                if i != j:
                    A[i][j] = 1.0
    return A


def test_sgtsbms_basic():
    """The two cliques are recovered exactly, and the summaries follow."""
    A = _two_cliques(3)
    r = sgt_sbm_spectral_estimate(A, k=2)
    labels = [int(v) for v in r["labels"]]
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert r["k"] == 2
    assert r["n"] == 6
    # every vertex has degree m - 1 = 2
    assert abs(float(r["mean_degree"]) - 2.0) < 1e-9
    assert abs(float(r["log_n"]) - math.log(6)) < 1e-9
    assert [int(s) for s in sorted(float(s) for s in r["block_sizes"])] == [3, 3]
    assert r["regularized"] is True


def test_sgtsbms_eigenvalues_of_two_cliques():
    """Two K_3 blocks: the adjacency spectrum is 2, 2, -1, -1, -1, -1.

    Unregularized, so the reported leading eigenvalues are the graph's own.
    """
    r = sgt_sbm_spectral_estimate(_two_cliques(3), k=2, regularized=False)
    ev = [float(v) for v in r["eigenvalues"]]
    assert abs(ev[0] - 2.0) < 1e-9
    assert abs(ev[1] - 2.0) < 1e-9
    assert abs(ev[2] + 1.0) < 1e-9
    # the eigengap separates the k-th from the (k+1)-th eigenvalue in magnitude
    assert abs(float(r["eigengap"]) - (abs(ev[1]) - abs(ev[2]))) < 1e-9
    assert r["regularized"] is False


def test_sgtsbms_edge():
    """A larger pair of blocks satisfies the degree condition."""
    r = sgt_sbm_spectral_estimate(_two_cliques(8), k=2)
    labels = [int(v) for v in r["labels"]]
    assert len(set(labels[:8])) == 1
    assert len(set(labels[8:])) == 1
    assert labels[0] != labels[8]
    assert abs(float(r["mean_degree"]) - 7.0) < 1e-9
    assert r["degree_condition"] is True


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.sgtsbms as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
