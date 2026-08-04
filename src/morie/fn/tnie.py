"""Total natural indirect effect."""

from __future__ import annotations

from ._richresult import RichResult
from .ctde import _effects, _fit

__all__ = ["total_natural_indirect_effect"]


def total_natural_indirect_effect(X, M, Y, C=None, a=1.0, astar=0.0):
    """TNIE = E[Y(a, M(a)) - Y(a, M(a*))] = (theta2 beta1 + theta3 beta1 a)(a - a*).

    Robins, J. M. and Greenland, S. (1992), "Identifiability and
    exchangeability for direct and indirect effects", *Epidemiology*
    3(2), 143-155, doi:10.1097/00001648-199203000-00013, is the shelf
    citation; it is closed access with no open copy in any repository
    (Unpaywall reports is_oa false).  This is the NIE printed as
    equation (0.3) of Valeri, L. and VanderWeele, T. J. (2013),
    *Psychological Methods* 18(2), 137-150, doi:10.1037/a0031034, open
    access at PMC3659198:

        NIE = (theta2 beta1 + theta3 beta1 a)(a - a*),

    the indirect effect with the exposure held at a in the outcome
    model.  Together with the pure natural direct effect it adds to the
    total effect, and the same total is reached the other way round by
    the total natural direct effect plus the pure natural indirect
    effect: PNDE + TNIE = TNDE + PNIE.  That identity is exact and is
    what this module is checked against.

    See ``controlled_direct_effect`` for the two fitted models and for
    the identification assumptions this arithmetic does not check.

    Parameters
    ----------
    X, M, Y : array-like
        Exposure, mediator, outcome.
    C : array-like or None
        Optional covariates.
    a, astar : float
        Exposure contrast.

    Returns
    -------
    estimate : TNIE
    pnde, tnde, pnie, te, mediated_interaction : the rest
    """
    beta, theta, cbar, n = _fit(X, M, Y, C, "total_natural_indirect_effect")
    eff = _effects(beta, theta, cbar, float(a), float(astar))
    out = dict(eff)
    out.update({
        "estimate": eff["tnie"],
        "a": float(a),
        "astar": float(astar),
        "n": n,
        "method": "Total natural indirect effect",
    })
    return RichResult(payload=out)


def cheatsheet():
    return "tnie: Total natural indirect effect"
