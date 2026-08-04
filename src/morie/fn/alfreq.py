# morie.fn -- function file (rootcoder007/morie)
"""Recycling embedder of AlphaFold."""

from __future__ import annotations

import math

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_recycling"]

#: The 15 recycling distance bins quoted in the supplement, 3 3/8 A to
#: 21 3/8 A.  See the note in :func:`alphafold_recycling` about their width.
RECYCLE_BINS = [3.375 + (21.375 - 3.375) / 14.0 * b for b in range(15)]


def alphafold_recycling(m1, z, x, wd, bins=None, ncycle=1):
    """Recycling embedder -- Algorithms 30 and 32, pp. 42-43.

    Recycling feeds the previous iteration's outputs back into the inputs:
    the pairwise distances between predicted beta-carbon positions are
    discretised, projected and added to the layer-normalised pair
    representation, and the first MSA row is layer-normalised.  This is the
    only channel through which a previous prediction reaches the network;
    everything else is recomputed from scratch each cycle.

    The iteration count is fixed and there is no tolerance-based early
    exit, so the result is deterministic.

    Parameters
    ----------
    m1 : list of list of float
        First row of the MSA representation, ``n x cm``.
    z : list of list of list of float
        Pair representation, ``n x n x cz``.
    x : list of list of float
        Predicted beta-carbon positions, ``n x 3`` (alpha carbon for
        glycine, as the spec notes).
    wd : list of list of float
        Projection of the one-hot distance encoding, ``cz x len(bins)``.
    bins : list of float, optional
        Distance bins; see the note below on the default.
    ncycle : int
        Number of recycling iterations to apply.  Algorithm 30 starts from
        zero-valued outputs, so the first iteration is well defined.

    Returns
    -------
    result : RichResult
        Keys: ``z`` (the update ``z~``), ``m1`` (the update ``m~1``),
        ``d`` (the distance matrix), ``estimate``, ``n``, ``method``.

    Notes
    -----
    The supplement describes these bins twice and the two descriptions do
    not agree: "15 bins of equal width 1.25 A" implies an upper bin at
    ``3.375 + 14 * 1.25 = 20.875``, while the parenthetical "precise bin
    values range from 3 3/8 A to 21 3/8 A" implies a width of
    ``18 / 14 = 1.2857``.  The default here follows the stated endpoints,
    since those are given as the precise values; pass ``bins`` explicitly
    for the other reading.

    The distance matrix is symmetric with a zero diagonal, so the
    contribution added to the pair representation is symmetric too.  The
    harness checks that rather than trusting the two arms to agree.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithms 30, 32
    """
    if bins is None:
        bins = RECYCLE_BINS
    n = len(z)
    cz = len(z[0][0])

    zc = [[list(z[i][j]) for j in range(n)] for i in range(n)]
    mc = [list(m1[i]) for i in range(n)]
    d = [[0.0] * n for _ in range(n)]
    for _ in range(ncycle):
        for i in range(n):
            for j in range(n):
                d[i][j] = math.sqrt(A.vnorm2(A.vsub(x[i], x[j])))
        zc = [[A.vadd(A.lin(A.onehotnb(d[i][j], bins), wd),
                      A.lnorm(zc[i][j])) for j in range(n)]
              for i in range(n)]
        mc = [A.lnorm(mc[i]) for i in range(n)]

    flat = [zc[i][j][t] for i in range(n) for j in range(n) for t in range(cz)]
    return RichResult(
        payload={
            "z": zc,
            "m1": mc,
            "d": d,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "ncycle": ncycle,
            "method": "AlphaFold recycling embedder",
        }
    )


def cheatsheet():
    return "alfreq: recycling embedder, distance histogram back into the pair"
