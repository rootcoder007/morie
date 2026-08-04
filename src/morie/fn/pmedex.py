# morie.fn -- function file (rootcoder007/morie)
"""Proportion of the total effect explained."""

from ._richresult import RichResult

__all__ = ["proportion_te_explained"]


def proportion_te_explained(nie, te):
    """Indirect effect divided by an independently supplied total.

    Not the same as the proportion mediated even though it usually
    prints the same number.  Here the total effect is whatever the
    caller estimated, which need not equal ``NIE + NDE`` when the two
    came from different models -- and when it does not, the gap is
    itself worth seeing, so the implied direct effect is returned.

    Formula: ``PTE = NIE / TE``.

    Parameters
    ----------
    nie : float
        Natural indirect effect.
    te : float
        Total effect, estimated separately.

    Returns
    -------
    RichResult
        ``estimate``, ``implied_nde`` (``te - nie``), ``te``.

    References
    ----------
    VanderWeele, T. J. (2013).  Policy-relevant proportions for direct
    effects.  Epidemiology 24:175-176.
    """
    nie = float(nie)
    te = float(te)
    return RichResult(payload={
        "estimate": nie / te if te != 0.0 else float("nan"),
        "implied_nde": te - nie, "te": te,
        "method": "Proportion of the total effect explained"})


def cheatsheet():
    return "pmedex: Proportion of the total effect explained."
