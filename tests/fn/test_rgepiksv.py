"""Tests for bsaclass.rangayyan_epilepsy_ksvd (Algorithm 9.2, sec. 9.8)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_epilepsy_ksvd


N = 8
BASIS = [[1.0 if i == j else 0.0 for i in range(N)] for j in range(N)]
SIG = [[5, 1, 0, 0, 0, 0, 3, 0], [4, 0, 2, 0, 0, 0, 0, 1],
       [0, 0, 0, 6, 0, 2, 0, 0], [0, 1, 0, 5, 3, 0, 0, 0]]
LAB = [0, 0, 1, 1]


def test_rgepiksv_basic():
    """With an orthonormal raw dictionary each pass of Algorithm 9.2 takes
    the atom of largest |<x, psi>|, i.e. the largest remaining coordinate,
    so two passes add each signal's two largest coordinates (deduplicated,
    in order).  Coefficients are <x, psi>, the reconstruction error is the
    norm of the coordinates left out, and classification is nearest
    class-mean feature vector."""
    r = rangayyan_epilepsy_ksvd(SIG, LAB, iterations=2, atoms=BASIS)
    order = []
    for s in SIG:
        for j in sorted(range(N), key=lambda i: -abs(s[i]))[:2]:
            if j not in order:
                order.append(j)
    assert r["dictionary"] == [BASIS[j] for j in order]
    for k, s in enumerate(SIG):
        assert r["coefficients"][k] == pytest.approx([s[j] for j in order], abs=1e-15)
        assert r["error"][k] == pytest.approx(math.sqrt(sum(s[i] ** 2 for i in range(N) if i not in order)), rel=1e-12)
    feats = [[s[j] for j in order] + [r["error"][k]] for k, s in enumerate(SIG)]
    cent = {c: [sum(f[m] for f, y in zip(feats, LAB) if y == c) / 2 for m in range(len(order) + 1)] for c in (0, 1)}
    pred = [min((0, 1), key=lambda c: sum((f[m] - cent[c][m]) ** 2 for m in range(len(f)))) for f in feats]
    assert r["predictions"] == pred == LAB
    assert r["accuracy"] == 1.0
    assert r["isseizure"] == [False, False, True, True]
    assert r["testclass"] is None


def test_rgepiksv_edge():
    """A single class, or mismatched signals and labels, raises."""
    with pytest.raises(ValueError):
        rangayyan_epilepsy_ksvd(SIG, [0, 0, 0, 0], atoms=BASIS)
    with pytest.raises(ValueError):
        rangayyan_epilepsy_ksvd(SIG, [0, 1], atoms=BASIS)
