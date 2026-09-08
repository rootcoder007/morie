# morie.fn -- function file (rootcoder007/morie)
"""Natural effects and their pure-plus-interaction refinement."""

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["causal_natural_decomposition"]


def causal_natural_decomposition(X, M, Y, Cc=None, a=1.0, astar=0.0, m=0.0):
    """Natural direct and indirect effects, then split each again.

    ``TE = NDE + NIE`` is where mediation analysis usually stops.  Each
    half still contains an interaction term, and separating them says
    something the two-way split cannot: whether the direct path carries
    an interaction with the mediator, and whether the indirect path
    does.  The pure indirect effect is the piece that would survive even
    if the exposure and mediator never interacted.

    Formula: ``NDE = CDE + INTref``, ``NIE = PIE + INTmed``,
    ``PDE = CDE + INTref``, and ``PIE`` as in the four-way
    decomposition; ``TE = NDE + NIE``.

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
        ``NDE``, ``NIE``, ``PDE``, ``PIE``, ``estimate`` (total
        effect), ``cde``, ``intref``, ``intmed``, ``n``.

    References
    ----------
    VanderWeele, T. J. (2015).  Explanation in Causal Inference:
    Methods for Mediation and Interaction.  Oxford University Press,
    chapter 14.  The algebra is that of VanderWeele (2014),
    Epidemiology 25:749-761, whose expressions were read directly and
    are reproduced in the four-way module of this package.
    """
    theta, beta, cbar = S.medmodels(Y, X, M, Cc)
    cde, intref, intmed, pie, te = S.fourway(theta, beta, cbar, a, astar, m)
    nde = cde + intref
    nie = pie + intmed
    return RichResult(payload={
        "NDE": nde, "NIE": nie, "PDE": nde, "PIE": pie, "estimate": te,
        "cde": cde, "intref": intref, "intmed": intmed, "n": len(C.vec(Y)),
        "method": "Natural effects with pure and interaction parts"})


def cheatsheet():
    return "causmnde: Natural effects and their pure-plus-interaction refinement."
