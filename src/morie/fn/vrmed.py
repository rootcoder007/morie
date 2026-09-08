# morie.fn -- function file (rootcoder007/morie)
"""Variance-based mediation share."""

from ._richresult import RichResult

__all__ = ["variance_based_mediation"]


def variance_based_mediation(r2_full, r2_partial):
    """Fraction of explained variance attributable to the mediator.

    A variance share is not a causal quantity, and this one in
    particular says nothing about direction -- it would report the same
    number if the arrow between mediator and outcome ran the other way.
    It is a descriptive complement to a coefficient-based proportion
    mediated, useful mainly because it is bounded in [0, 1] when the
    partial model is nested in the full one.

    Formula: ``R2_med = (R2_full - R2_partial) / R2_full``.

    Parameters
    ----------
    r2_full : float
        R-squared of the model including the mediator.
    r2_partial : float
        R-squared of the model without it.

    Returns
    -------
    RichResult
        ``estimate``, ``delta_r2``, ``r2_full``, ``r2_partial``.

    References
    ----------
    de Heus, P. (2012).  R squared effect-size measures and overlap
    between direct and indirect effect in mediation analysis.
    Behavior Research Methods 44:213-221.
    """
    rf = float(r2_full)
    rp = float(r2_partial)
    return RichResult(payload={
        "estimate": (rf - rp) / rf if rf != 0.0 else float("nan"),
        "delta_r2": rf - rp, "r2_full": rf, "r2_partial": rp,
        "method": "Variance-based mediation share"})


def cheatsheet():
    return "vrmed: Variance-based mediation share."
