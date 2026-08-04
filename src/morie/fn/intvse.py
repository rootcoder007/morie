# morie.fn -- function file (rootcoder007/morie)
"""Interventional direct and indirect effects."""

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["interventional_effect"]


def interventional_effect(Y, X, M, Cc=None, a=1.0, astar=0.0):
    """Effects defined by shifting the mediator distribution, not the value.

    Natural effects need each person counterfactual mediator value
    under the exposure they did not get, which is not identified when
    something the exposure caused also confounds the mediator-outcome
    link.  Interventional effects sidestep that by drawing the mediator
    from its population distribution under the other exposure level
    instead of from the individual.  They are identified under weaker
    assumptions, and they answer a question a policy could actually
    implement -- at the cost of no longer decomposing an individual
    effect.

    Formula: with ``G_a`` a random draw from the distribution of ``M``
    given exposure ``a`` and covariates,
    ``IDE = (th1 + th3 (b0 + b1 a* + b2' c))(a - a*)`` and
    ``IIE = (th2 + th3 a) b1 (a - a*)``, which sum to the total effect.

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
        ``estimate`` (total), ``ide``, ``iie``, ``check`` (total minus
        the two parts), ``theta``, ``beta``, ``n``.

    References
    ----------
    VanderWeele, T. J., Vansteelandt, S. & Robins, J. M. (2014).
    Effect decomposition in the presence of an exposure-induced
    mediator-outcome confounder.  Epidemiology 25:300-306.  The
    randomised-interventional analogues are defined there; the linear
    expressions above follow from the same outcome and mediator models
    used in VanderWeele (2014), Epidemiology 25:749-761, which was read
    directly.
    """
    theta, beta, cbar = S.medmodels(Y, X, M, Cc)
    d = a - astar
    bc = beta[0] + beta[1] * astar + sum(beta[2 + j] * cbar[j] for j in range(len(cbar)))
    ide = (theta[1] + theta[3] * bc) * d
    iie = (theta[2] + theta[3] * a) * beta[1] * d
    cde, intref, intmed, pie, te = S.fourway(theta, beta, cbar, a, astar, 0.0)
    return RichResult(payload={
        "estimate": te, "ide": ide, "iie": iie, "check": te - (ide + iie),
        "theta": theta, "beta": beta, "n": len(C.vec(Y)),
        "method": "Interventional direct and indirect effects"})


def cheatsheet():
    return "intvse: Interventional direct and indirect effects."
