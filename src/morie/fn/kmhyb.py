# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid retrieval: weighted fusion of dense and sparse scores."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_hybrid_retrieval_fusion"]


def kamath_hybrid_retrieval_fusion(s_dense, s_sparse, lam, normalize=False):
    """score = lam * s_dense + (1 - lam) * s_sparse, ranked descending.

    Dense and sparse scores live on different scales (inner products
    vs BM25), so ``normalize=True`` min-max maps each arm to [0, 1]
    before fusing; the default does NOT, because silently rescaling a
    caller's scores changes the ranking they asked for. Ties keep the
    lower document index.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 7, hybrid retrieval.

    Examples
    --------
    >>> out = kamath_hybrid_retrieval_fusion([1.0, 0.0], [0.0, 1.0], 0.75)
    >>> out["scores"]
    [0.75, 0.25]
    >>> out["ranking"]
    [0, 1]
    """
    d = np.atleast_1d(np.asarray(s_dense, dtype=float)).ravel()
    s = np.atleast_1d(np.asarray(s_sparse, dtype=float)).ravel()
    lam = float(lam)
    if d.size != s.size:
        raise ValueError(
            f"the two arms score different numbers of documents: "
            f"{d.size} dense vs {s.size} sparse.")
    if d.size == 0:
        raise ValueError("no documents to fuse.")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must lie in [0, 1]; got {lam}.")
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(s))):
        raise ValueError("scores must be finite.")

    def mm(v):
        lo, hi = v.min(), v.max()
        if hi == lo:
            raise ValueError(
                "an arm gives every document the same score, so min-max "
                "normalisation is 0/0; fuse the raw scores instead.")
        return (v - lo) / (hi - lo)

    dd, ss = (mm(d), mm(s)) if normalize else (d, s)
    fused = lam * dd + (1.0 - lam) * ss
    order = np.argsort(-fused, kind="stable")
    return RichResult(payload={
        "scores": [float(v) for v in fused],
        "ranking": [int(i) for i in order],
        "estimate": float(fused[order[0]]),
        "lam": lam, "normalized": bool(normalize),
        "n": int(fused.size),
        "method": "Hybrid dense/sparse score fusion"})


def cheatsheet():
    return "kmhyb: lam*dense + (1-lam)*sparse, ranked; optional min-max"
