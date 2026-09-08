# morie.fn -- function file (rootcoder007/morie)
"""Distogram prediction head of AlphaFold."""

from __future__ import annotations

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_distogram"]


def distogram_bins(nbins=64, lo=2.0, hi=22.0):
    """Bin centres covering ``lo`` to ``hi`` in equal widths.

    The spec's last bin also absorbs every larger distance; that only
    matters when binning a target, which ``onehotnb`` handles by nearest
    centre.
    """
    wdt = (hi - lo) / (nbins - 1)
    return [lo + wdt * b for b in range(nbins)]


def alphafold_distogram(z, w, bins=None, dtrue=None):
    """Distogram prediction -- supplement section 1.9.8, p. 39.

    The pair representation is symmetrised, projected to distance bins and
    passed through a softmax.  Symmetrising is the whole trick: a distance
    is a symmetric quantity, and forcing the head to see ``z_ij + z_ji``
    guarantees the prediction respects that regardless of what the trunk
    produced.

    Parameters
    ----------
    z : list of list of list of float
        Pair representation, ``n x n x cz``.
    w : list of list of float
        Projection to the bin logits, ``nbins x cz``.
    bins : list of float, optional
        Bin centres; defaults to 64 bins from 2 to 22 angstrom.
    dtrue : list of list of float, optional
        Ground-truth distances.  When given, the averaged cross-entropy of
        equation (41) is returned as well.

    Returns
    -------
    result : RichResult
        Keys: ``p`` (bin distributions), ``dist`` (expected distance),
        ``loss`` (or ``None``), ``estimate``, ``n``, ``method``.

    Notes
    -----
    Two properties anchor this and the harness checks both: the output is
    exactly symmetric in ``i`` and ``j``, and each distribution sums to
    one, so the expected distance lies within the bin range and equals the
    mean of the bins when the weights are zero.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary section 1.9.8
    """
    if bins is None:
        bins = distogram_bins()
    n = len(z)
    cz = len(z[0][0])
    nb = len(bins)
    ps, dd = [], []
    for i in range(n):
        prow, drow = [], []
        for j in range(n):
            sym = [z[i][j][t] + z[j][i][t] for t in range(cz)]
            p = A.smax(A.lin(sym, w))
            prow.append(p)
            drow.append(sum(p[b] * bins[b] for b in range(nb)))
        ps.append(prow)
        dd.append(drow)

    loss = None
    if dtrue is not None:
        loss = sum(A.xent(A.onehotnb(dtrue[i][j], bins), ps[i][j])
                   for i in range(n) for j in range(n)) / (n * n)

    flat = [dd[i][j] for i in range(n) for j in range(n)]
    return RichResult(
        payload={
            "p": ps,
            "dist": dd,
            "loss": loss,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "method": "AlphaFold distogram prediction",
        }
    )


def cheatsheet():
    return "alfdst: distogram head over the symmetrised pair representation"
