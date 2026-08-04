# morie.fn -- function file (rootcoder007/morie)
"""Per-residue model confidence (pLDDT) head of AlphaFold."""

from __future__ import annotations

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_confidence"]

#: Bin centres of the pLDDT head, ``[1, 3, 5, ..., 99]`` (Algorithm 29).
PLDDT_BINS = [1.0 + 2.0 * b for b in range(50)]


def alphafold_confidence(s, w1, w2, w3, bins=None, rtrue=None):
    """Predict per-residue lDDT-Ca -- Algorithm 29, p. 37.

    A small MLP on the single representation produces a distribution over
    lDDT bins; the reported confidence is that distribution's mean.  This
    is the quantity AlphaFold reports as pLDDT.

    All weights are supplied by the caller.

    Parameters
    ----------
    s : list of list of float
        Single representation, ``n x cs``.
    w1, w2 : list of list of float
        The two projections of line 1, each followed by a relu.
    w3 : list of list of float
        Projection to the bin logits, ``nbins x c`` (line 2).
    bins : list of float, optional
        Bin centres.  Defaults to the spec's ``[1, 3, ..., 99]``.
    rtrue : list of float, optional
        Ground-truth lDDT per residue.  When given, the cross-entropy of
        line 4 is returned as well.

    Returns
    -------
    result : RichResult
        Keys: ``plddt`` (per-residue confidence, line 5), ``p`` (the bin
        distributions), ``loss`` (the confidence loss, or ``None``),
        ``estimate`` (mean pLDDT), ``n``, ``method``.

    Notes
    -----
    Line 4 of the published pseudocode reads ``Lconf = mean(ptrue' log p)``
    with no leading minus sign, which would make the "loss" something to be
    maximised while equation (7) adds it to a total that is minimised.  The
    reference implementation uses the negative log likelihood, so that is
    what is computed here; the sign in the supplement is a typo.

    Two closed forms anchor this: the bin distribution sums to one, and the
    reported confidence is a convex combination of the bin centres, so with
    zero final weights the logits are flat and the answer is exactly the
    mean of the bins.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithm 29
    """
    if bins is None:
        bins = PLDDT_BINS
    n = len(s)
    ps, r = [], []
    for i in range(n):
        a = [A.relu(t) for t in A.lin(A.lnorm(s[i]), w1)]
        a = [A.relu(t) for t in A.lin(a, w2)]
        p = A.smax(A.lin(a, w3))
        ps.append(p)
        r.append(sum(p[b] * bins[b] for b in range(len(bins))))

    loss = None
    if rtrue is not None:
        loss = sum(A.xent(A.onehotnb(rtrue[i], bins), ps[i])
                   for i in range(n)) / n

    return RichResult(
        payload={
            "plddt": r,
            "p": ps,
            "loss": loss,
            "estimate": sum(r) / n,
            "n": n,
            "method": "AlphaFold per-residue confidence (pLDDT)",
        }
    )


def cheatsheet():
    return "alfcnf: per-residue confidence head, pLDDT"
