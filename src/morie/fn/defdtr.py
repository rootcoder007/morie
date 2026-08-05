# morie.fn -- function file (rootcoder007/morie)
"""Deformable DETR sparse attention."""

import math

from . import _array_core as np
from . import _s03core as core
from ._richresult import RichResult

__all__ = ["deformable_detr"]


def bilinear(F, y, x):
    """Bilinear sample of a feature map at a real-valued location."""
    H = len(F)
    W = len(F[0])
    y = min(max(y, 0.0), H - 1.0)
    x = min(max(x, 0.0), W - 1.0)
    y0 = int(math.floor(y))
    x0 = int(math.floor(x))
    y1 = min(y0 + 1, H - 1)
    x1 = min(x0 + 1, W - 1)
    dy = y - y0
    dx = x - x0
    return (F[y0][x0] * (1 - dy) * (1 - dx) + F[y0][x1] * (1 - dy) * dx
            + F[y1][x0] * dy * (1 - dx) + F[y1][x1] * dy * dx)


def deformable_detr(x, queries, K=4, offsets=None, weights=None, seed=42):
    """
    Deformable DETR attention

    Formula: K reference points per query; sparse attention

    Each query attends to only K sampled locations around its reference
    point instead of every position, so the cost is O(K) rather than
    O(HW) and convergence no longer needs 500 epochs.  The samples are
    read by bilinear interpolation, which is what makes the learned
    offsets differentiable.  With zero offsets and equal weights the
    output is exactly the feature at the reference point -- the
    degenerate case that pins the interpolation.

    Parameters
    ----------
    x : array-like
        H x W feature map.
    queries : array-like
        Q x 2 matrix of reference points in normalised [0, 1] coords.
    K : int
        Sampling points per query.
    offsets : array-like or None
        Q x K x 2 sampling offsets in pixels; None draws them from the
        deterministic stream.
    weights : array-like or None
        Q x K attention weights; None uses 1/K each.
    seed : int
        Seed of the deterministic stream.

    Returns
    -------
    result : dict
        Keys: estimate (mean output), out, samples, ref_pixels, Q, K.

    References
    ----------
    Zhu et al. (2021), Deformable DETR: Deformable Transformers for
    End-to-End Object Detection, ICLR 2021.
    """
    F = core.mat(x)
    H = len(F)
    if H == 0:
        raise ValueError("empty input: x has no rows")
    W = len(F[0])
    Qm = core.mat(queries)
    Q = len(Qm)
    if Q == 0 or len(Qm[0]) != 2:
        raise ValueError("queries must be a Q x 2 matrix of reference points")
    K = int(K)
    if K < 1:
        raise ValueError("K must be at least 1")
    rng = np.random.default_rng(seed)
    if offsets is None:
        off = [[[float(rng.normal(0.0, 1.0)) for _ in range(2)]
                for _ in range(K)] for _ in range(Q)]
    else:
        flat = core.vec(offsets)
        if len(flat) != Q * K * 2:
            raise ValueError("offsets must hold Q x K x 2 values")
        off = [[[flat[(q * K + k) * 2 + c] for c in range(2)]
                for k in range(K)] for q in range(Q)]
    if weights is None:
        wt = [[1.0 / K] * K for _ in range(Q)]
    else:
        flat = core.vec(weights)
        if len(flat) != Q * K:
            raise ValueError("weights must hold Q x K values")
        wt = [[flat[q * K + k] for k in range(K)] for q in range(Q)]
    out, samples, refs = [], [], []
    for q in range(Q):
        ry = Qm[q][1] * (H - 1)
        rx = Qm[q][0] * (W - 1)
        refs.append([ry, rx])
        s = 0.0
        row = []
        tot = sum(wt[q])
        for k in range(K):
            v = bilinear(F, ry + off[q][k][0], rx + off[q][k][1])
            row.append(v)
            s += wt[q][k] * v
        samples.append(row)
        out.append(s / tot if tot != 0.0 else 0.0)
    return RichResult(payload={
        "estimate": sum(out) / Q,
        "out": out,
        "samples": samples,
        "ref_pixels": refs,
        "Q": Q,
        "K": K,
        "method": "Deformable DETR sparse attention",
    })


def cheatsheet():
    return "defdtr: Deformable DETR sparse attention"


# compact alias per ledger/NAMING.md
deformabledetr = deformable_detr
