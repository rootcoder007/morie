# morie.fn -- function file (rootcoder007/morie)
r"""Survey-weighted quantile regression.

The mean is the wrong summary when the question is about the tails: a
policy that moves the median earner and one that moves the 90th
percentile are different policies, and a conditional mean cannot tell
them apart. Quantile regression minimises the asymmetric check loss
:math:`\rho_\tau(u) = u(\tau - 1\{u<0\})`, so the fit at
:math:`\tau = 0.9` is genuinely about the upper tail rather than about
the mean plus an assumption.

With survey data the loss carries the DESIGN WEIGHTS,
:math:`\sum_i w_i \rho_\tau(y_i - x_i'\beta)`, because the sample is not
the population -- an unweighted quantile of a stratified sample is a
quantile of the wrong distribution. Minimisation here is by
majorise-minimise: the check loss is bounded above by a weighted
quadratic that touches it at the current fit, so each iteration is one
weighted least-squares solve and the objective cannot increase.

The objective is returned each iteration for exactly that reason -- a
monotone decrease is the property that makes the algorithm trustworthy,
and it is checkable rather than assertable.

References
----------
Koenker, R. (2005) *Quantile Regression*, Econometric Society Monograph
38, Cambridge University Press, Ch. 1-2 (the check function, the
asymmetric-loss characterisation of quantiles) and Ch. 6 (computation).

Koenker, R. and Bassett, G. (1978) "Regression quantiles",
*Econometrica* **46**(1), 33-50, doi:10.2307/1913643.

Lumley, T. (2010) *Complex Surveys: A Guide to Analysis Using R*, Wiley,
Ch. 2 (why design weights belong in the estimating equation).

Hunter, D. R. and Lange, K. (2000) "Quantile regression via an MM
algorithm", *Journal of Computational and Graphical Statistics* **9**(1),
60-77, doi:10.1080/10618600.2000.10474866. The majorise-minimise
algorithm used here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["survey_quantile_regression"]

_EPS = 1e-12


def survey_quantile_regression(X, y, tau=0.5, weights=None,
                               add_intercept=True, max_iter=200,
                               tol=1e-10, eps=1e-6):
    r"""Minimise sum_i w_i rho_tau(y_i - x_i'beta) by majorise-minimise."""
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    yv = [float(v) for v in k.vec(y)]
    n = len(Xm)
    if n == 0:
        raise ValueError("svyrcq: no observations")
    if len(yv) != n:
        raise ValueError("svyrcq: %d rows but %d responses" % (n, len(yv)))
    tau = float(tau)
    if not 0.0 < tau < 1.0:
        raise ValueError("svyrcq: tau must lie strictly in (0, 1), got %r"
                         % (tau,))
    if weights is None:
        w = [1.0] * n
    else:
        w = [float(v) for v in k.vec(weights)]
        if len(w) != n:
            raise ValueError("svyrcq: %d rows but %d weights" % (n, len(w)))
        if any(v < 0.0 for v in w):
            raise ValueError("svyrcq: design weights cannot be negative")
    if add_intercept:
        Xm = [[1.0] + r for r in Xm]
    p = len(Xm[0])
    if n <= p:
        raise ValueError("svyrcq: %d observations cannot identify %d "
                         "coefficients" % (n, p))

    def check_loss(beta):
        s = 0.0
        for i in range(n):
            u = yv[i] - sum(Xm[i][a] * beta[a] for a in range(p))
            s += w[i] * u * (tau - (1.0 if u < 0.0 else 0.0))
        return s

    def wls(om, adj):
        A = [[sum(om[i] * Xm[i][a] * Xm[i][b] for i in range(n))
              for b in range(p)] for a in range(p)]
        scale = sum(A[a][a] for a in range(p)) / p
        ridge = 1e-10 * scale if scale > _EPS else 1e-12
        for a in range(p):
            A[a][a] += ridge
        b = [sum(Xm[i][a] * (om[i] * yv[i] + adj[i]) for i in range(n))
             for a in range(p)]
        return k.cholsolve(A, b)

    beta = wls([wi for wi in w], [0.0] * n)      # weighted LS start
    obj = [check_loss(beta)]
    it, converged = 0, False
    for it in range(1, int(max_iter) + 1):
        om, adj = [], []
        for i in range(n):
            u = yv[i] - sum(Xm[i][a] * beta[a] for a in range(p))
            d = max(abs(u), eps)
            om.append(w[i] / (2.0 * d))
            adj.append(w[i] * (tau - 0.5))
        new = wls(om, adj)
        shift = max(abs(new[a] - beta[a]) for a in range(p))
        beta = new
        obj.append(check_loss(beta))
        if shift < tol:
            converged = True
            break

    res = [yv[i] - sum(Xm[i][a] * beta[a] for a in range(p))
           for i in range(n)]
    below = sum(w[i] for i in range(n) if res[i] < 0.0)
    tot = sum(w)
    return RichResult(payload={
        "estimate": beta, "coefficients": beta, "residuals": res,
        "fitted": [yv[i] - res[i] for i in range(n)],
        "objective": obj[-1], "objective_path": obj,
        "weighted_fraction_below": below / tot if tot > _EPS else 0.0,
        "tau": tau, "iterations": it, "converged": converged,
        "n": n, "p": p, "sum_weights": tot,
        "method": "survey-weighted quantile regression by majorise-minimise "
                  "(Koenker 2005; Hunter & Lange 2000; Lumley 2010)",
        "note": "the majorising quadratic touches the check loss at the "
                "current fit, so the objective cannot increase -- "
                "objective_path makes that checkable rather than asserted",
    })


def cheatsheet():
    return ("svyrcq: survey_quantile_regression(X, y, tau, weights) -> "
            "design-weighted quantile regression by MM (Koenker 2005, "
            "Quantile Regression, CUP; Lumley 2010, Complex Surveys)")
