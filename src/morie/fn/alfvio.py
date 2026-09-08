# morie.fn -- function file (rootcoder007/morie)
"""Structural violation losses of AlphaFold."""

from __future__ import annotations

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_violation"]


def _flat(x, tol):
    """Flat-bottom L1: nothing is charged inside the tolerance."""
    e = abs(x) - tol
    return e if e > 0.0 else 0.0


def alphafold_violation(blen=None, blen_lit=None, blen_sigma=None,
                        cosang=None, cosang_lit=None, cosang_sigma=None,
                        dnb=None, dnb_lit=None, factor=12.0, clash_tol=1.5):
    """Structural violation loss -- supplement section 1.9.11, equations
    (44)-(47), p. 40.

    Three flat-bottom penalties that charge nothing while the geometry
    stays within tolerance and grow linearly beyond it: bond lengths
    against literature values, bond angles through the cosine of the angle,
    and a one-sided clash term on non-bonded pairs that penalises only
    distances that are too short.

    The tolerances follow the spec: ``factor`` (12 by default) times the
    literature standard deviation for bonds and angles, and a flat 1.5 A
    for clashes.

    Parameters
    ----------
    blen, blen_lit, blen_sigma : list of float, optional
        Predicted bond lengths, their literature values and standard
        deviations (equation 44).
    cosang, cosang_lit, cosang_sigma : list of float, optional
        Cosines of predicted and literature bond angles, and the
        literature standard deviations (equation 45).
    dnb, dnb_lit : list of float, optional
        Predicted distances between non-bonded atom pairs and their
        clashing distances (equation 46).
    factor : float
        Multiplier on the literature standard deviation, 12 in the spec.
    clash_tol : float
        Clash tolerance in angstrom, 1.5 in the spec.

    Returns
    -------
    result : RichResult
        Keys: ``bondlength``, ``bondangle``, ``clash``, ``estimate`` (their
        sum, equation 47), ``method``.

    Notes
    -----
    Equations (44) and (45) average over bonds and angles while (46) sums
    over non-bonded pairs; that asymmetry is in the published text and is
    reproduced here rather than tidied away.

    Two closed forms anchor this and the harness checks both: the loss is
    exactly zero when every quantity sits inside its tolerance, and beyond
    the tolerance it is exactly linear, so doubling every excess doubles
    the loss.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary eq. (44)-(47)
    """
    lb = 0.0
    if blen is not None:
        lb = sum(_flat(blen[i] - blen_lit[i], factor * blen_sigma[i])
                 for i in range(len(blen))) / len(blen)
    la = 0.0
    if cosang is not None:
        la = sum(_flat(cosang[i] - cosang_lit[i], factor * cosang_sigma[i])
                 for i in range(len(cosang))) / len(cosang)
    lc = 0.0
    if dnb is not None:
        # one-sided: only distances shorter than the clashing distance
        lc = sum(max(dnb_lit[i] - clash_tol - dnb[i], 0.0)
                 for i in range(len(dnb)))

    return RichResult(
        payload={
            "bondlength": lb,
            "bondangle": la,
            "clash": lc,
            "estimate": lb + la + lc,
            "method": "AlphaFold structural violation loss",
        }
    )


def cheatsheet():
    return "alfvio: flat-bottom bond length, bond angle and clash penalties"
