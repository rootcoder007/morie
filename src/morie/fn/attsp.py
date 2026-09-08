# morie.fn -- function file (rootcoder007/morie)
"""Sparse (masked) scaled dot-product attention."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sparseattn", "sparse_attention"]


def sparseattn(Q, K, V, S=None):
    """Attention restricted to a fixed connectivity pattern.

    Full self-attention costs O(n^2) because every query attends to every
    key.  Sparse transformers keep the same arithmetic but let each query
    attend only to a prescribed subset of positions, given by a 0/1
    connectivity matrix S:

        Attn(Q, K, V) = softmax( mask_S( Q K' / sqrt(d) ) ) V

    where mask_S sets the score to -infinity wherever S is zero, so the
    softmax gives those positions exactly zero weight.  The pattern
    itself -- strided, fixed, dilated, global -- is the design choice;
    supplying S separates that choice from the arithmetic.

    Parameters
    ----------
    Q : array-like, shape (nq, d)
        Queries.
    K : array-like, shape (nk, d)
        Keys.
    V : array-like, shape (nk, dv)
        Values.
    S : array-like, shape (nq, nk) or None
        Connectivity mask; non-zero means attendable.  ``None`` is dense
        attention.

    Returns
    -------
    RichResult
        ``out``, ``weight``, ``score``, ``nq``, ``nk``, ``d``, ``dv``,
        ``density``.

    References
    ----------
    Child, R., Gray, S., Radford, A. and Sutskever, I. (2019),
    "Generating long sequences with sparse transformers",
    arXiv:1904.10509, Sect. 3-4, which replaces the full connectivity of
    attention with fixed strided patterns while leaving the scaled
    dot-product softmax untouched; a copy of the paper is in the local
    corpus.  Beltagy, I., Peters, M. E. and Cohan, A. (2020),
    "Longformer: the long-document transformer", arXiv:2004.05150, is the
    windowed-plus-global variant of the same masking.  The scaled
    dot-product form itself is Vaswani et al. (2017), Equation (1).
    """
    Qm = C.mat(Q)
    Km = C.mat(K)
    Vm = C.mat(V)
    nq, d = len(Qm), len(Qm[0])
    nk = len(Km)
    if len(Km[0]) != d:
        raise ValueError("Q and K must share their last dimension")
    if len(Vm) != nk:
        raise ValueError("V must have one row per key")
    dv = len(Vm[0])
    if S is None:
        Sm = [[1.0] * nk for _ in range(nq)]
    else:
        Sm = C.mat(S)
        if len(Sm) != nq or len(Sm[0]) != nk:
            raise ValueError("S must be nq by nk")
    sc = math.sqrt(d)
    sco = [[sum(Qm[i][t] * Km[j][t] for t in range(d)) / sc
            for j in range(nk)] for i in range(nq)]
    Wt = []
    for i in range(nq):
        allow = [j for j in range(nk) if Sm[i][j] != 0.0]
        if not allow:
            raise ValueError("row %d of S allows no key" % i)
        mx = max(sco[i][j] for j in allow)
        e = [0.0] * nk
        tot = 0.0
        for j in allow:
            e[j] = math.exp(sco[i][j] - mx)
            tot += e[j]
        Wt.append([v / tot for v in e])
    out = [[sum(Wt[i][j] * Vm[j][t] for j in range(nk)) for t in range(dv)]
           for i in range(nq)]
    dens = sum(1 for i in range(nq) for j in range(nk)
               if Sm[i][j] != 0.0) / float(nq * nk)
    return RichResult(payload={
        "out": out, "weight": Wt, "score": sco, "nq": nq, "nk": nk,
        "d": d, "dv": dv, "density": dens,
        "method": "Sparse scaled dot-product attention (Child et al. 2019)"})


sparse_attention = sparseattn


def cheatsheet():
    return "attsp: Sparse (masked) scaled dot-product attention."
