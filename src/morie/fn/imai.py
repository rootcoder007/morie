# morie.fn -- function file (rootcoder007/morie)
"""Imai-Keele-Yamamoto causal mediation."""

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["imai_keele_yamamoto_mediation"]


def imai_keele_yamamoto_mediation(X, M, Y, Cc=None):
    """Average causal mediation effects under sequential ignorability.

    The contribution here was not a new estimator but a clean statement
    of what has to be true: sequential ignorability, meaning the
    exposure is as good as random given covariates and the mediator is
    as good as random given exposure and covariates.  Everything else,
    including the familiar product of coefficients, follows -- and when
    exposure and mediator interact, the mediation effect depends on
    which exposure level you evaluate it at, so both are returned rather
    than one being quietly chosen.

    Formula: with ``Y = th0 + th1 a + th2 m + th3 a m + th4' c`` and
    ``M = b0 + b1 a + b2' c``, the mediation effect is
    ``delta(a) = b1 (th2 + th3 a)`` and the direct effect is
    ``zeta(a) = th1 + th3 (b0 + b1 a + b2' c)``.

    Parameters
    ----------
    X : array-like, shape (n,)
        Treatment.
    M : array-like, shape (n,)
        Mediator.
    Y : array-like, shape (n,)
        Outcome.
    Cc : array-like, optional
        Covariates; read at their means.

    Returns
    -------
    RichResult
        ``estimate`` (average of the two mediation effects), ``acme_0``,
        ``acme_1``, ``ade_0``, ``ade_1``, ``total``, ``prop_mediated``,
        ``n``.

    References
    ----------
    Imai, K., Keele, L. & Yamamoto, T. (2010).  Identification,
    inference and sensitivity analysis for causal mediation effects.
    Statistical Science 25:51-71, equations (7) and (8), which are the
    quantities the authors own mediation package reports.
    """
    theta, beta, cbar = S.medmodels(Y, X, M, Cc)
    bc0 = beta[0] + sum(beta[2 + j] * cbar[j] for j in range(len(cbar)))
    d0 = beta[1] * theta[2]
    d1 = beta[1] * (theta[2] + theta[3])
    z0 = theta[1] + theta[3] * bc0
    z1 = theta[1] + theta[3] * (bc0 + beta[1])
    total = 0.5 * (d0 + d1) + 0.5 * (z0 + z1)
    return RichResult(payload={
        "estimate": 0.5 * (d0 + d1), "acme_0": d0, "acme_1": d1,
        "ade_0": z0, "ade_1": z1, "total": total,
        "prop_mediated": 0.5 * (d0 + d1) / total if total != 0.0 else float("nan"),
        "n": len(C.vec(Y)),
        "method": "Imai-Keele-Yamamoto causal mediation"})


def cheatsheet():
    return "imai: Imai-Keele-Yamamoto causal mediation."
