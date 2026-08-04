# morie.fn -- function file (rootcoder007/morie)
"""Four-way decomposition of a total effect."""

from . import _s04core as S

from ._richresult import RichResult

__all__ = ["interaction_mediation_4way"]


def interaction_mediation_4way(X, M, Y, Cc=None, a=1.0, astar=0.0, m=0.0):
    """Split a total effect into mediation, interaction, both, neither.

    Mediation analysis and interaction analysis had been separate
    literatures asking overlapping questions.  The four-way
    decomposition shows they are two halves of one accounting identity:
    every total effect divides into a part due to neither (the
    controlled direct effect), a part due to interaction alone, a part
    due to mediation alone, and a part that needs both at once.  The
    fourth piece is the one neither literature could name on its own.

    Formula: ``TE = CDE + INTref + INTmed + PIE``, evaluated from a
    linear outcome model ``Y = th0 + th1 a + th2 m + th3 a m + th4' c``
    and mediator model ``M = b0 + b1 a + b2' c``.

    Parameters
    ----------
    X : array-like, shape (n,)
        Exposure.
    M : array-like, shape (n,)
        Mediator.
    Y : array-like, shape (n,)
        Outcome.
    Cc : array-like, optional
        Covariates; the decomposition is read at their means.
    a, astar : float
        Exposure contrast levels.
    m : float, default 0.0
        Level the mediator is controlled at for the CDE.

    Returns
    -------
    RichResult
        ``estimate`` (the total effect), ``cde``, ``intref``,
        ``intmed``, ``pie``, ``pai`` (portion attributable to
        interaction), ``pe`` (portion eliminated), ``theta``, ``beta``,
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
    cde, intref, intmed, pie, te = S.fourway(theta, beta, cbar, a, astar, m)
    return RichResult(payload={
        "estimate": te, "cde": cde, "intref": intref, "intmed": intmed,
        "pie": pie, "pai": intref + intmed, "pe": intref + intmed + pie,
        "theta": theta, "beta": beta, "n": len(list(Y)),
        "method": "VanderWeele four-way decomposition"})


def cheatsheet():
    return "intmd4: Four-way decomposition of a total effect."
