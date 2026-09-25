"""Tests for vitfsv.vit_finetune (ViT downstream head, Dosovitskiy et al. 2021)."""

import math

import pytest

from morie.fn.vitfsv import vit_finetune


X = [[math.sin(0.9 * i + 0.4 * j) + (1.5 if (i % 3) == j else 0.0) for j in range(3)] for i in range(12)]
Y = [(i % 3) + 1 for i in range(12)]


def _solve(A, B):
    n = len(A)
    M = [A[i][:] + B[i][:] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [[M[i][n + k] / M[i][i] for k in range(len(B[0]))] for i in range(n)]


def test_vitfsv_basic():
    """Few-shot head (sec. 4.1): W = (X'X + lambda I)^-1 X'T with T the
    {-1, 1}^K targets; logits X W; prediction the first maximum."""
    lam = 0.3
    T = [[1.0 if y == k + 1 else -1.0 for k in range(3)] for y in Y]
    XtX = [[sum(r[a] * r[b] for r in X) + (lam if a == b else 0.0) for b in range(3)] for a in range(3)]
    XtT = [[sum(r[a] * t[k] for r, t in zip(X, T)) for k in range(3)] for a in range(3)]
    W = _solve(XtX, XtT)
    r = vit_finetune(X, Y, mode="fewshot", ridge=lam)
    assert [list(map(float, row)) for row in r["head"]] == [pytest.approx(row, abs=1e-12) for row in W]
    logits = [[sum(x[a] * W[a][k] for a in range(3)) for k in range(3)] for x in X]
    pred = [max(range(3), key=lambda k: (lg[k], -k)) + 1 for lg in logits]
    assert list(r["pred"]) == pred
    assert r["estimate"] == pytest.approx(sum(p == y for p, y in zip(pred, Y)) / 12, abs=1e-15)


def test_vitfsv_edge():
    """The zero-initialised head (sec. 3.2) gives all-zero logits and,
    by the first-maximum rule, class 1 everywhere; full fine-tuning is
    refused; labels outside 1..K raise."""
    r = vit_finetune(X, Y, mode="init")
    assert list(r["pred"]) == [1] * 12
    assert r["estimate"] == pytest.approx(4 / 12, abs=1e-15)
    with pytest.raises((ValueError, NotImplementedError)):
        vit_finetune(X, Y, mode="full")
    with pytest.raises(ValueError):
        vit_finetune(X, [0] * 12)
