# morie.fn -- function file (rootcoder007/morie)
"""Mediated interaction."""

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mediated_interaction"]


def mediated_interaction(Y, X, M, Cc=None, a=1.0, astar=0.0):
    """The part of the effect that needs mediation and interaction at once.

    This is the component neither literature could express alone.  It is
    non-zero only when the exposure both moves the mediator and
    interacts with it in the outcome, so a single zero in either place
    kills it -- which makes it a sharp diagnostic rather than a residual
    catch-all.

    Formula: ``INTmed = th3 b1 (a - a*)^2``; the nonparametric analogue
    for binary variables is
    ``(p11 - p10 - p01 + p00){P(M = 1|A = 1) - P(M = 1|A = 0)}``, the
    empirical column of the paper table 1.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    X : array-like, shape (n,)
        Exposure.
    M : array-like, shape (n,)
        Mediator.
    Cc : array-like, optional
        Covariates; read at their means.
    a, astar : float
        Exposure contrast levels.

    Returns
    -------
    RichResult
        ``estimate`` (INTmed), ``interaction`` (``th3``),
        ``mediator_shift`` (``b1 (a - a*)``), ``theta``, ``beta``,
        ``n``.

    References
    ----------
    VanderWeele, T. J. (2014).  A unification of mediation and
    interaction: a 4-way decomposition.  Epidemiology 25:749-761.
    Fetched and read; the regression expressions used here are printed
    in that paper as
    ``E[CDE|c] = th1 (a - a*)``,
    ``E[INTref|c] = th3 (b0 + b1 a* + b2'c)(a - a*)``,
    ``E[INTmed|c] = th3 b1 (a - a*)(a - a*)`` and
    ``E[PIE|c] = (th2 b1 + th3 b1 a*)(a - a*)``.
    """
    theta, beta, cbar = S.medmodels(Y, X, M, Cc)
    d = a - astar
    return RichResult(payload={
        "estimate": theta[3] * beta[1] * d * d, "interaction": theta[3],
        "mediator_shift": beta[1] * d, "theta": theta, "beta": beta,
        "n": len(C.vec(Y)), "method": "Mediated interaction INTmed"})


def cheatsheet():
    return "medint: Mediated interaction."
