# morie.fn -- function file (rootcoder007/morie)
"""Gravity model of flows by Poisson pseudo-maximum likelihood (Santos Silva and Tenreyro 2006)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, ssum
from ._richresult import RichResult

__all__ = ["gravity_ppml"]


def gravity_ppml(flows, mass_o, mass_d, dist, *, tol: float = 1e-12, max_iter: int = 100) -> RichResult:
    r"""Gravity model ``E[F_od] = exp(b0 + b1 log M_o + b2 log M_d + b3 log d_od)`` by PPML.

    The Poisson quasi-likelihood score equations are solved by iteratively
    reweighted least squares (a Poisson GLM with log link, ``glm(...,
    family = poisson)``); PPML is consistent under heteroskedasticity and
    keeps zero flows (Santos Silva and Tenreyro 2006).  Standard errors are
    the Eicker-White sandwich ``(X'MX)^{-1} X' diag((F - mu)^2) X (X'MX)^{-1}``
    (HC0), the usual choice for PPML; the model-based (Poisson) ones are also
    returned, with the log-likelihood, AIC and BIC.

    :param flows: Non-negative flows (n,).
    :param mass_o: Origin masses (n,), positive.
    :param mass_d: Destination masses (n,), positive.
    :param dist: Distances (n,), positive.
    :return: :class:`RichResult` with ``coefficients`` (intercept, origin,
        destination, distance elasticities), ``se_robust``, ``se_model``,
        ``fitted``, ``loglik``, ``aic``, ``bic``, ``iterations``.

    References
    ----------
    Santos Silva, J. M. C. and Tenreyro, S. (2006). The log of gravity.
    *Review of Economics and Statistics*, 88(4), 641-658.

    Examples
    --------
    >>> F = [12.0, 0.0, 30.0, 7.0, 55.0, 3.0]
    >>> r = gravity_ppml(F, [5, 5, 9, 9, 20, 20], [9, 20, 5, 20, 5, 9], [1.0, 3.0, 1.0, 2.0, 3.0, 2.0])
    >>> [round(v, 6) for v in r.coefficients]
    [8.8362, -0.610957, -2.50985, 0.884215]
    """
    F = [float(v) for v in np.asarray(flows, dtype=float).tolist()]
    mo = [float(v) for v in np.asarray(mass_o, dtype=float).tolist()]
    md = [float(v) for v in np.asarray(mass_d, dtype=float).tolist()]
    d = [float(v) for v in np.asarray(dist, dtype=float).tolist()]
    n = len(F)
    if not (len(mo) == len(md) == len(d) == n):
        raise ValueError("all inputs must have the same length")
    if min(F) < 0 or min(mo) <= 0 or min(md) <= 0 or min(d) <= 0:
        raise ValueError("flows must be non-negative and masses and distances positive")
    X = [[1.0, math.log(mo[i]), math.log(md[i]), math.log(d[i])] for i in range(n)]
    p = 4
    mu = [(F[i] + sum(F) / n) / 2.0 for i in range(n)]
    eta = [math.log(m) for m in mu]
    b = [0.0] * p
    dev_old = float("inf")
    it = 0
    for step in range(1, max_iter + 1):
        it = step
        z = [eta[i] + (F[i] - mu[i]) / mu[i] for i in range(n)]
        XtW = [[X[i][a] * mu[i] for i in range(n)] for a in range(p)]
        A = [[ssum(XtW[a][i] * X[i][c] for i in range(n)) for c in range(p)] for a in range(p)]
        rhs = [ssum(XtW[a][i] * z[i] for i in range(n)) for a in range(p)]
        Ai = [[float(v) for v in r] for r in inverse(A)]
        b = [ssum(Ai[a][c] * rhs[c] for c in range(p)) for a in range(p)]
        eta = [ssum(X[i][a] * b[a] for a in range(p)) for i in range(n)]
        mu = [math.exp(v) for v in eta]
        dev = 2.0 * ssum((F[i] * math.log(F[i] / mu[i]) if F[i] > 0 else 0.0) - (F[i] - mu[i]) for i in range(n))
        if abs(dev - dev_old) / (abs(dev) + 0.1) < tol:
            break
        dev_old = dev
    XtW = [[X[i][a] * mu[i] for i in range(n)] for a in range(p)]
    Ai = [
        [float(v) for v in r]
        for r in inverse([[ssum(XtW[a][i] * X[i][c] for i in range(n)) for c in range(p)] for a in range(p)])
    ]
    meat = [[ssum(X[i][a] * X[i][c] * (F[i] - mu[i]) ** 2 for i in range(n)) for c in range(p)] for a in range(p)]
    Vr = [
        [ssum(Ai[a][k] * meat[k][m] * Ai[m][c] for k in range(p) for m in range(p)) for c in range(p)] for a in range(p)
    ]
    ll = ssum(F[i] * eta[i] - mu[i] - math.lgamma(F[i] + 1.0) for i in range(n))
    return RichResult(
        payload={
            "coefficients": b,
            "se_robust": [Vr[k][k] ** 0.5 for k in range(p)],
            "se_model": [Ai[k][k] ** 0.5 for k in range(p)],
            "fitted": mu,
            "loglik": ll,
            "aic": -2.0 * ll + 2.0 * p,
            "bic": -2.0 * ll + p * math.log(n),
            "iterations": it,
        }
    )


def cheatsheet() -> str:
    return "gravity_ppml(flows, mass_o, mass_d, dist) -> PPML gravity elasticities (Santos Silva-Tenreyro)."


# alias kept from the retired placeholder of the same name
migration_flow = gravity_ppml
