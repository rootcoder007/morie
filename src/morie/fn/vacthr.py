# morie.fn -- function file (rootcoder007/morie)
"""Critical vaccination threshold for herd immunity."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['vaccthresh', 'vaccination_threshold']


def vaccthresh(R0, efficacy=1.0):
    """Critical vaccination threshold for herd immunity.

    The threshold is the fraction that must be immune, not the fraction that must be vaccinated: an imperfect vaccine needs coverage p_c/e, which exceeds one -- meaning herd immunity is unreachable by vaccination alone -- as soon as e < 1 - 1/R0. That case is reported rather than clipped away.


    Formula: p_c = 1 - 1/R0; with vaccine efficacy e the coverage needed is p_c/e

    Parameters
    ----------
    R0 : float or array-like
        Basic reproduction number, greater than 1.
    efficacy : float
        Vaccine efficacy in (0, 1].

    Returns
    -------
    RichResult
        ``threshold``, ``coverage``, ``feasible``, ``R0``, ``efficacy``.

    References
    ----------
    Anderson and May (1991), Infectious Diseases of Humans: Dynamics and
    Control, Oxford University Press.  Not held locally; p_c = 1 - 1/R0 is
    the standard published result and is stated in the same form in every
    open source consulted.
    """
    r0 = C.vec(R0)
    e = float(efficacy)
    if e <= 0 or e > 1:
        raise ValueError("efficacy must be in (0, 1]")
    if any(v <= 0 for v in r0):
        raise ValueError("R0 must be positive")
    pc = [1.0 - 1.0 / v for v in r0]
    cov = [p / e for p in pc]
    return RichResult(payload={
        "threshold": pc if len(pc) > 1 else pc[0],
        "coverage": cov if len(cov) > 1 else cov[0],
        "feasible": [c <= 1.0 for c in cov] if len(cov) > 1 else cov[0] <= 1.0,
        "R0": r0 if len(r0) > 1 else r0[0], "efficacy": e,
        "method": "Critical vaccination threshold"})


vaccination_threshold = vaccthresh


def cheatsheet():
    return "vacthr: Critical vaccination threshold for herd immunity."
