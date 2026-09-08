# morie.fn -- function file (rootcoder007/morie)
"""Predicted aligned error (PAE) head of AlphaFold."""

from __future__ import annotations

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_pae_predict"]

#: Bin centres of the PAE head: 64 bins of width 0.5 A from 0 to 31.5 A.
PAE_BINS = [0.25 + 0.5 * b for b in range(64)]


def alphafold_pae_predict(z, w, bins=None):
    """Predicted aligned error -- supplement section 1.9.7, p. 38.

    The error in the position of residue ``j`` when the prediction and the
    ground truth are aligned on residue ``i`` is predicted as a
    distribution over distance bins, obtained by projecting the pair
    representation and taking a softmax.  The reported PAE is that
    distribution's mean.

    Unlike the distogram head this one does **not** symmetrise the pair
    representation: aligning on ``i`` and reading off ``j`` is a different
    question from the reverse, so the resulting matrix is deliberately
    non-symmetric.

    Parameters
    ----------
    z : list of list of list of float
        Pair representation, ``n x n x cz``.
    w : list of list of float
        Projection to the bin logits, ``nbins x cz``.
    bins : list of float, optional
        Bin centres.  Defaults to the spec's 64 bins of width 0.5 A
        covering 0 to 31.5 A.

    Returns
    -------
    result : RichResult
        Keys: ``pae`` (the ``n x n`` expected error), ``p`` (the bin
        distributions), ``estimate`` (mean PAE), ``n``, ``method``.

    Notes
    -----
    Each distribution sums to one, and the expected error is a convex
    combination of the bin centres, so zero weights give exactly the mean
    of the bins.  Both are checked in the parity harness.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary section 1.9.7
    """
    if bins is None:
        bins = PAE_BINS
    n = len(z)
    nb = len(bins)
    pae, ps = [], []
    for i in range(n):
        prow, erow = [], []
        for j in range(n):
            p = A.smax(A.lin(z[i][j], w))
            prow.append(p)
            erow.append(sum(p[b] * bins[b] for b in range(nb)))
        ps.append(prow)
        pae.append(erow)

    flat = [pae[i][j] for i in range(n) for j in range(n)]
    return RichResult(
        payload={
            "pae": pae,
            "p": ps,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "method": "AlphaFold predicted aligned error",
        }
    )


def cheatsheet():
    return "alfpea: predicted aligned error head, non-symmetric by design"
