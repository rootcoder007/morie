"""Controlled direct effect (Robins-Greenland)."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["controlled_direct_effect"]


def _fit(X, M, Y, C, who):
    """The two regressions of the regression-based mediation formulas.

    Mediator:  E[M | a, c] = b0 + b1 a + b2' c
    Outcome:   E[Y | a, m, c] = t0 + t1 a + t2 m + t3 a m + t4' c
    """
    a = core.vec(X)
    m = core.vec(M)
    y = core.vec(Y)
    n = len(a)
    if n == 0:
        raise ValueError(who + ": X is empty")
    if len(m) != n or len(y) != n:
        raise ValueError(who + ": X, M and Y must have the same length")
    if C is None:
        cols = []
    else:
        cm = core.mat(C)
        if len(cm) != n:
            raise ValueError(who + ": C must have one row per observation")
        cols = [[cm[i][j] for i in range(n)] for j in range(len(cm[0]))]
    if n < 4 + len(cols):
        raise ValueError(who + ": too few observations to fit both models")
    dm = [[1.0, a[i]] + [cl[i] for cl in cols] for i in range(n)]
    dy = [[1.0, a[i], m[i], a[i] * m[i]] + [cl[i] for cl in cols] for i in range(n)]
    beta = core.lstsq(dm, m)
    theta = core.lstsq(dy, y)
    cbar = [sum(cl) / n for cl in cols]
    return beta, theta, cbar, n


def _effects(beta, theta, cbar, a, astar):
    """Valeri and VanderWeele (2013), eq. (0.3), and its two mirror images."""
    d = a - astar
    b0 = beta[0]
    b1 = beta[1]
    bc = 0.0
    for j in range(2, len(beta)):
        bc += beta[j] * cbar[j - 2]
    t1 = theta[1]
    t2 = theta[2]
    t3 = theta[3]
    pnde = (t1 + t3 * (b0 + b1 * astar + bc)) * d
    tnde = (t1 + t3 * (b0 + b1 * a + bc)) * d
    tnie = (t2 * b1 + t3 * b1 * a) * d
    pnie = (t2 * b1 + t3 * b1 * astar) * d
    return {"pnde": pnde, "tnde": tnde, "tnie": tnie, "pnie": pnie,
            "te": pnde + tnie, "mediated_interaction": t3 * b1 * d * d,
            "beta": beta, "theta": theta}


def controlled_direct_effect(X, M, Y, m, C=None, a=1.0, astar=0.0):
    """CDE(m) = (theta1 + theta3 m)(a - a*).

    Robins, J. M. and Greenland, S. (1992), "Identifiability and
    exchangeability for direct and indirect effects", *Epidemiology*
    3(2), 143-155, doi:10.1097/00001648-199203000-00013, is the shelf
    citation and where the controlled and natural effects are defined;
    it is closed access with no open copy in any repository (Unpaywall
    reports is_oa false, oa_locations empty).  The regression-based
    identification used here was read instead from an open source that
    states it in closed form, Valeri, L. and VanderWeele, T. J. (2013),
    "Mediation analysis allowing for exposure-mediator interactions and
    causal interpretation", *Psychological Methods* 18(2), 137-150,
    doi:10.1037/a0031034, open access at PMC3659198, equation (0.3),
    with a the new exposure level and a* the baseline one:

        CDE  = (theta1 + theta3 m)(a - a*)
        NDE  = {theta1 + theta3 (beta0 + beta1 a* + beta2' c)}(a - a*)
        NIE  = (theta2 beta1 + theta3 beta1 a)(a - a*)

    from the mediator model E[M|a,c] = beta0 + beta1 a + beta2' c and
    the outcome model
    E[Y|a,m,c] = theta0 + theta1 a + theta2 m + theta3 a m + theta4' c,
    both fitted here by ordinary least squares.  Valeri and VanderWeele's
    NDE is the pure one (mediator held at its a* distribution) and their
    NIE is the total one; the two mirror images, TNDE and PNIE, follow by
    swapping a and a* and are returned as well.

    The causal reading of these numbers needs the identification
    assumptions of that paper (no unmeasured exposure-outcome,
    mediator-outcome or exposure-mediator confounding, and no
    mediator-outcome confounder affected by the exposure).  This
    function does the arithmetic; it cannot check them.

    Parameters
    ----------
    X : array-like
        Exposure.
    M : array-like
        Mediator.
    Y : array-like
        Outcome.
    m : float
        Level at which the mediator is controlled.
    C : array-like or None
        Optional matrix of covariates, one row per observation.
    a, astar : float
        Exposure contrast; the default is 1 versus 0.

    Returns
    -------
    estimate : CDE(m)
    pnde, tnde, tnie, pnie, te : the rest of the decomposition
    beta, theta : the two fitted coefficient vectors
    """
    beta, theta, cbar, n = _fit(X, M, Y, C, "controlled_direct_effect")
    eff = _effects(beta, theta, cbar, float(a), float(astar))
    mm = float(m)
    out = dict(eff)
    out.update({
        "estimate": (theta[1] + theta[3] * mm) * (float(a) - float(astar)),
        "m": mm,
        "a": float(a),
        "astar": float(astar),
        "n": n,
        "method": "Controlled direct effect (Robins-Greenland)",
    })
    return RichResult(payload=out)


def cheatsheet():
    return "ctde: Controlled direct effect (Robins-Greenland)"
