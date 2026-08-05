# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Kullback-Leibler divergence with additive smoothing for sparse counts.

Chen and Goodman (1996), "An empirical study of smoothing techniques
for language modeling", Proceedings of the 34th Annual Meeting of the
Association for Computational Linguistics, pp. 310-318,
doi:10.3115/981863.981904, section 2.1: additive (Lidstone) smoothing
replaces a maximum-likelihood count ratio by

    P(w) = (c(w) + eps) / (sum_w c(w) + V eps),

with V the vocabulary size.  Without it the Kullback-Leibler divergence
of two sparse count vectors is infinite the moment the second assigns
zero to an event the first does not, which is the failure mode the
smoothing exists to prevent.  Both count vectors are smoothed with the
same eps and the divergence

    KL(P || Q) = sum_w P(w) log( P(w) / Q(w) )

is then finite by construction.  The reverse divergence and their sum
(Kullback's symmetric divergence J) are reported as well, since KL is
not symmetric.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["kl_molecular_smooth"]


def kl_molecular_smooth(p, q, eps):
    """Additively smoothed KL divergence between two count vectors.

    Parameters
    ----------
    p, q : array-like
        Non-negative counts (or weights) over the same vocabulary.
    eps : float
        Additive smoothing constant, strictly positive.
    """
    pv = core.vec(p)
    qv = core.vec(q)
    V = len(pv)
    if V == 0:
        raise ValueError("kl_molecular_smooth: p is empty")
    if len(qv) != V:
        raise ValueError("kl_molecular_smooth: p and q have different lengths")
    for v in pv + qv:
        if v < 0:
            raise ValueError("kl_molecular_smooth: counts must be non-negative")
    e = float(eps)
    if e <= 0:
        raise ValueError("kl_molecular_smooth: eps must be positive")
    sp = sum(pv) + V * e
    sq = sum(qv) + V * e
    P = [(v + e) / sp for v in pv]
    Q = [(v + e) / sq for v in qv]
    kl = 0.0
    rk = 0.0
    for i in range(V):
        kl += P[i] * math.log(P[i] / Q[i])
        rk += Q[i] * math.log(Q[i] / P[i])
    nz_p = sum(1 for v in pv if v == 0.0)
    nz_q = sum(1 for v in qv if v == 0.0)
    return RichResult(
        title="KL divergence with additive smoothing",
        summary_lines=[("vocabulary", V), ("KL(P||Q)", kl), ("eps", e)],
        payload={
            "estimate": kl,
            "kl_pq": kl,
            "kl_qp": rk,
            "symmetric_kl": kl + rk,
            "eps": e,
            "vocabulary": float(V),
            "zeros_p": float(nz_p),
            "zeros_q": float(nz_q),
            "mass_p": sum(pv),
            "mass_q": sum(qv),
            "n": V,
            "method": "P = (c + eps)/(N + V eps) then KL(P||Q) = sum P log(P/Q), Chen & Goodman (1996)",
        },
    )


def cheatsheet():
    return "klmsm1: KL with smoothing for sparse"


# compact alias per ledger/NAMING.md
klmolecularsmooth = kl_molecular_smooth
