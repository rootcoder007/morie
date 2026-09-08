# morie.fn -- function file (rootcoder007/morie)
"""TM-score structural alignment measure."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tm_score"]


def tm_score(coords1, coords2, l_ref=None):
    """Structure similarity whose scale does not drift with chain length.

    RMSD has no fixed meaning across lengths -- 3 angstrom is excellent
    for a 300-residue protein and poor for a 40-residue one.  The
    ``d_0`` normalisation is what fixes that, so a TM-score above 0.5
    means the same thing at any size: the same fold.  The distance
    weighting also caps the damage a few badly placed loops can do,
    which RMSD cannot.

    Formula: ``TM = (1 / L_ref) sum_i 1 / (1 + (d_i / d_0)^2)`` with
    ``d_0 = 1.24 (L_ref - 15)^(1/3) - 1.8``.

    Parameters
    ----------
    coords1, coords2 : array-like, shape (L, 3)
        Aligned residue coordinates; already superposed.
    l_ref : int, optional
        Reference length; the number of aligned residues by default.

    Returns
    -------
    RichResult
        ``estimate`` (TM-score), ``d0``, ``rmsd``, ``L``, ``L_ref``.

    References
    ----------
    Zhang, Y. & Skolnick, J. (2004).  Scoring function for automated
    assessment of protein structure template quality.  Proteins
    57:702-710, equations (2) and (3).
    """
    A = C.mat(coords1)
    B = C.mat(coords2)
    L = len(A)
    Lr = float(l_ref) if l_ref is not None else float(L)
    d0 = 1.24 * (Lr - 15.0) ** (1.0 / 3.0) - 1.8 if Lr > 15.0 else 0.5
    tot = 0.0
    ss = 0.0
    for i in range(L):
        d2 = sum((A[i][k] - B[i][k]) ** 2 for k in range(3))
        ss += d2
        tot += 1.0 / (1.0 + d2 / (d0 * d0))
    return RichResult(payload={
        "estimate": tot / Lr, "d0": d0, "rmsd": math.sqrt(ss / L),
        "L": L, "L_ref": Lr, "method": "TM-score structural similarity"})


tmscore = tm_score


def cheatsheet():
    return "tmscore: TM-score structural alignment measure."
