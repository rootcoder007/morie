# morie.fn -- function file (rootcoder007/morie)
"""Initial representation embeddings of AlphaFold."""

from __future__ import annotations

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_embedding_init"]

#: Relative position bins, ``[-32, ..., 32]`` (Algorithm 4).
RELPOS_BINS = [float(b) for b in range(-32, 33)]


def alphafold_embedding_init(target_feat, residue_index, msa_feat, wa, wb,
                             wrel, wmsa, wtgt, bins=None):
    """Embeddings for the initial representations -- Algorithms 3-5, p. 13.

    The pair representation starts as an outer sum of two projections of
    the target features plus a relative position encoding; the MSA
    representation starts as a projection of the MSA features with the
    target projection broadcast across sequences.

    The relative position encoding (Algorithm 4) clips the separation at
    32 residues, so nothing beyond that is distinguished.  That is
    deliberate: it de-emphasises primary sequence distance and lets the
    network run on chains longer than those it was trained on.

    All weights are supplied by the caller.

    Parameters
    ----------
    target_feat : list of list of float
        Per-residue target features, ``n x ctf``.
    residue_index : list of float
        Residue index per position; only differences matter.
    msa_feat : list of list of list of float
        MSA features, ``s x n x cmf``.
    wa, wb : list of list of float
        The two target projections of line 1, each ``cz x ctf``.
    wrel : list of list of float
        Relative position projection, ``cz x len(bins)`` (Algorithm 4).
    wmsa : list of list of float
        MSA feature projection, ``cm x cmf`` (line 4).
    wtgt : list of list of float
        Target projection added to every MSA row, ``cm x ctf`` (line 4).
    bins : list of float, optional
        Relative position bins; defaults to ``[-32, ..., 32]``.

    Returns
    -------
    result : RichResult
        Keys: ``z`` (initial pair representation), ``m`` (initial MSA
        representation), ``pos`` (the relative position contribution),
        ``estimate``, ``n``, ``method``.

    Notes
    -----
    Two structural properties anchor this and the harness checks both.  The
    relative position term depends only on ``i - j``, so its contribution
    is constant along each diagonal.  And before that term is added the
    pair representation is a pure outer sum ``a_i + b_j``, so the second
    difference ``z_ij - z_ik - z_lj + z_lk`` vanishes identically.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithms 3-5
    """
    if bins is None:
        bins = RELPOS_BINS
    n = len(target_feat)
    a = [A.lin(target_feat[i], wa) for i in range(n)]
    b = [A.lin(target_feat[i], wb) for i in range(n)]
    cz = len(a[0])

    pos, z = [], []
    for i in range(n):
        prow, zrow = [], []
        for j in range(n):
            d = residue_index[i] - residue_index[j]
            # Algorithm 5: one-hot with nearest bin, which is what clips
            # the separation at the ends of the bin range
            p = A.lin(A.onehotnb(d, bins), wrel)
            prow.append(p)
            zrow.append([a[i][t] + b[j][t] + p[t] for t in range(cz)])
        pos.append(prow)
        z.append(zrow)

    s = len(msa_feat)
    m = [[A.vadd(A.lin(msa_feat[si][i], wmsa), A.lin(target_feat[i], wtgt))
          for i in range(n)] for si in range(s)]

    flat = [z[i][j][t] for i in range(n) for j in range(n) for t in range(cz)]
    return RichResult(
        payload={
            "z": z,
            "m": m,
            "pos": pos,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "method": "AlphaFold initial representation embeddings",
        }
    )


def cheatsheet():
    return "alfemb: initial pair and MSA embeddings with relative positions"
