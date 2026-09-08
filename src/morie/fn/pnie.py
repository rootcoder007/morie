"""Pure natural indirect effect."""

from __future__ import annotations

from ._richresult import RichResult
from .ctde import _effects, _fit

__all__ = ["pure_natural_indirect_effect"]


def pure_natural_indirect_effect(X, M, Y, C=None, a=1.0, astar=0.0):
    """PNIE = E[Y(a*, M(a)) - Y(a*, M(a*))] = (theta2 beta1 + theta3 beta1 a*)(a - a*).

    Robins, J. M. and Greenland, S. (1992), "Identifiability and
    exchangeability for direct and indirect effects", *Epidemiology*
    3(2), 143-155, doi:10.1097/00001648-199203000-00013, is the shelf
    citation; it is closed access with no open copy in any repository
    (Unpaywall reports is_oa false).  The closed form is the mirror
    image of the NIE of equation (0.3) of Valeri, L. and VanderWeele,
    T. J. (2013), *Psychological Methods* 18(2), 137-150,
    doi:10.1037/a0031034, open access at PMC3659198, which gives the
    total natural indirect effect (theta2 beta1 + theta3 beta1 a)(a - a*);
    the pure one holds the exposure at a* in the outcome model instead,
    so a becomes a*.  The two differ by theta3 beta1 (a - a*)^2, the
    mediated interaction, which is returned as well and is exactly zero
    when the outcome model carries no exposure-mediator interaction.

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
    estimate : PNIE
    pnde, tnde, tnie, te, mediated_interaction : the rest
    """
    beta, theta, cbar, n = _fit(X, M, Y, C, "pure_natural_indirect_effect")
    eff = _effects(beta, theta, cbar, float(a), float(astar))
    out = dict(eff)
    out.update({
        "estimate": eff["pnie"],
        "a": float(a),
        "astar": float(astar),
        "n": n,
        "method": "Pure natural indirect effect",
    })
    return RichResult(payload=out)


def cheatsheet():
    return "pnie: Pure natural indirect effect"
