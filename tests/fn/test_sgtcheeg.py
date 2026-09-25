"""Tests for sgtcheeg.sgt_cheeger_constant."""

import itertools
import math

import pytest

from morie.fn.sgtcheeg import sgt_cheeger_constant

# two triangles joined by the bridge 2-3
A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]


def _h_exact():
    deg = [sum(r) for r in A]
    vol = sum(deg)
    best = math.inf
    for k in range(1, 6):
        for S in itertools.combinations(range(6), k):
            s = set(S)
            cut = sum(A[i][j] for i in s for j in range(6) if j not in s)
            vs = sum(deg[i] for i in s)
            best = min(best, cut / min(vs, vol - vs))
    return best


def test_sgtcheeg_basic():
    """The exact constant by exhaustive search over all 62 cuts is the
    bridge, 1 / vol(triangle) = 1/7; the Fiedler sweep finds it, and
    lambda_2 / 2 <= h <= sqrt(2 lambda_2) (Cheeger's inequality)."""
    h = _h_exact()
    assert h == pytest.approx(1 / 7, rel=1e-15)
    r = sgt_cheeger_constant(A)
    assert r["sweep_min"] == pytest.approx(h, rel=1e-12)
    assert r["lower_bound"] <= h <= r["upper_bound"]
    assert sorted(r["cut_set"]) in ([0, 1, 2], [3, 4, 5])


def test_sgtcheeg_edge():
    """An isolated vertex makes conductance undefined and is refused."""
    with pytest.raises(ValueError):
        sgt_cheeger_constant([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
