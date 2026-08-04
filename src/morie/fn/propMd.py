# morie.fn -- function file (rootcoder007/morie)
"""Proportion mediated."""

from ._richresult import RichResult

__all__ = ["proportion_mediated"]


def proportion_mediated(NIE, NDE):
    """Share of the total effect that runs through the mediator.

    The measure only reads as a proportion when the two pieces point the
    same way.  With opposite signs the denominator is a difference of
    magnitudes and the ratio can exceed one or go negative -- which is
    information, not an error, so it is returned along with a flag
    rather than clamped.

    Formula: ``PM = NIE / (NIE + NDE)``.

    Parameters
    ----------
    NIE : float
        Natural indirect effect.
    NDE : float
        Natural direct effect.

    Returns
    -------
    RichResult
        ``estimate``, ``te``, ``same_sign`` (1 when both components
        share a sign, so the ratio reads as a proportion).

    References
    ----------
    VanderWeele, T. J. (2013).  Policy-relevant proportions for direct
    effects.  Epidemiology 24:175-176, and VanderWeele (2015),
    Explanation in Causal Inference, Oxford University Press, section
    2.7, where the caveat about opposite signs is stated.
    """
    nie = float(NIE)
    nde = float(NDE)
    te = nie + nde
    same = 1.0 if (nie >= 0.0) == (nde >= 0.0) else 0.0
    return RichResult(payload={
        "estimate": nie / te if te != 0.0 else float("nan"), "te": te,
        "same_sign": same, "method": "Proportion mediated, NIE / TE"})


def cheatsheet():
    return "propMd: Proportion mediated."
