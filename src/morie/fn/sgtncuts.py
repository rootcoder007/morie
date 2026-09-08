# morie.fn -- slice s03 (rootcoder007/morie)
"""Normalised cut of a partition.

Source consulted: Shi, J. and Malik, J. (2000).  Normalized cuts and
image segmentation.  *IEEE Transactions on Pattern Analysis and Machine
Intelligence* 22(8), 888-905, whose equation (2) is

    Ncut(A, B) = cut(A, B) / assoc(A, V) + cut(A, B) / assoc(B, V)

with cut(A, B) = sum_(u in A, v in B) w(u, v) and assoc(A, V) =
sum_(u in A, t in V) w(u, t) -- so it is the cut weight charged against
the total *volume* of each side, which is what stops the criterion
preferring to shave off a single vertex.  The PAMI paper is paywalled;
the equation is quoted in its standard published form.  The companion
normalised association, Nassoc(A, B), satisfies Ncut = 2 - Nassoc for a
two-way cut, and is returned so the identity can be checked.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["sgt_normalised_cut"]


def sgt_normalised_cut(A, labels):
    """Ncut of a two-way (or k-way) partition of a weighted graph.

    Parameters
    ----------
    A : 2-D array-like
        Symmetric weight matrix.
    labels : array-like
        Partition label per node.

    Returns
    -------
    RichResult with payload:
        ncut     : the normalised cut
        estimate : same as ncut
        cut      : total inter-group weight
        vol      : volume of each group
        nassoc   : the normalised association
    """
    W = k.mat(A)
    n = len(W)
    lab = [str(x) for x in labels]
    ids = []
    for c in lab:
        if c not in ids:
            ids.append(c)
    vol = []
    for c in ids:
        s = 0.0
        for i in range(n):
            if lab[i] == c:
                for j in range(n):
                    s += W[i][j]
        vol.append(s)
    ncut = 0.0
    nassoc = 0.0
    cut_total = 0.0
    for gi, c in enumerate(ids):
        cut = 0.0
        assoc = 0.0
        for i in range(n):
            if lab[i] != c:
                continue
            for j in range(n):
                if lab[j] != c:
                    cut += W[i][j]
                else:
                    assoc += W[i][j]
        cut_total += cut
        if vol[gi] > 0.0:
            ncut += cut / vol[gi]
            nassoc += assoc / vol[gi]
    return RichResult(
        title="Normalised cut",
        summary_lines=[("Ncut", ncut), ("groups", len(ids))],
        payload={
            "ncut": ncut,
            "estimate": ncut,
            "cut": cut_total / 2.0,
            "vol": vol,
            "nassoc": nassoc,
            "n_groups": len(ids),
            "method": "Normalised cut (Shi and Malik 2000, eq. 2)",
        },
    )


def cheatsheet():
    return "sgtncuts: Normalised cut objective for a partition"
