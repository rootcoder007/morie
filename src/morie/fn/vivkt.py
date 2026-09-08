# morie.fn -- function file (rootcoder007/morie)
"""Two-way and four-way decompositions side by side."""

from . import _s04core as S

from ._richresult import RichResult

__all__ = ["vansteelandt_vanderweele"]


def vansteelandt_vanderweele(X, M, Y, Cc=None, a=1.0, astar=0.0, m=0.0):
    """Show how the classical two-way split sits inside the four-way one.

    The Robins-Greenland-Pearl decomposition into a natural direct and a
    natural indirect effect is not a rival of the four-way split; it is
    the four-way split with two pairs already summed.  ``CDE + INTref``
    is the pure direct effect and ``PIE + INTmed`` the total indirect
    effect, so the older decomposition is recovered exactly and the
    interaction it was hiding becomes visible.

    Formula: ``PDE = CDE + INTref``, ``TIE = PIE + INTmed``, and
    ``TE = PDE + TIE = CDE + INTref + INTmed + PIE``.

    Parameters
    ----------
    X : array-like, shape (n,)
        Exposure.
    M : array-like, shape (n,)
        Mediator.
    Y : array-like, shape (n,)
        Outcome.
    Cc : array-like, optional
        Covariates; read at their means.
    a, astar, m : float
        Exposure contrast and controlled mediator level.

    Returns
    -------
    RichResult
        ``estimate`` (total effect), ``pde``, ``tie``, ``cde``,
        ``intref``, ``intmed``, ``pie``, ``check`` (the two totals
        differenced, which is zero to rounding), ``n``.

    References
    ----------
    VanderWeele, T. J. (2014).  A unification of mediation and
    interaction: a 4-way decomposition.  Epidemiology 25:749-761.
    Fetched and read; the regression expressions used here are printed
    in that paper as
    ``E[CDE|c] = th1 (a - a*)``,
    ``E[INTref|c] = th3 (b0 + b1 a* + b2'c)(a - a*)``,
    ``E[INTmed|c] = th3 b1 (a - a*)(a - a*)`` and
    ``E[PIE|c] = (th2 b1 + th3 b1 a*)(a - a*)``.  The two-way split it recovers is Robins, J. M. & Greenland, S.
    (1992), Identifiability and exchangeability for direct and indirect
    effects, Epidemiology 3:143-155, and Pearl, J. (2001), Direct and
    indirect effects, UAI 17:411-420.
    """
    theta, beta, cbar = S.medmodels(Y, X, M, Cc)
    cde, intref, intmed, pie, te = S.fourway(theta, beta, cbar, a, astar, m)
    pde = cde + intref
    tie = pie + intmed
    return RichResult(payload={
        "estimate": te, "pde": pde, "tie": tie, "cde": cde, "intref": intref,
        "intmed": intmed, "pie": pie, "check": te - (pde + tie),
        "n": len(list(Y)),
        "method": "Two-way and four-way decompositions together"})


def cheatsheet():
    return "vivkt: Two-way and four-way decompositions side by side."
