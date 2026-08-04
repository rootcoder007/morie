# morie.fn -- function file (rootcoder007/morie)
"""Weibull hazard model with unobserved heterogeneity: Honore estimator.

Horowitz, J. L. (2009), Semiparametric and Nonparametric Methods in
Econometrics, Springer, Section 6.1.3, pages 197-200 (volume
[Pages 189-232], read as rendered page images).  The model is

    lambda(y | x) = alpha y^(alpha - 1) exp(-x beta - V)      (6.17)
    alpha log Y   = X beta + U + V                            (6.18)

with V independent of (X, U) and otherwise unrestricted.  The book notes
below (6.19) why alpha cannot be recovered from the moment condition
(6.7): it holds for any value of alpha once gamma = beta / alpha is
free.  Honore (1990) instead recovers alpha from the behaviour of the
survivor function at the origin,

    alpha = lim_{y -> 0} log{-log[1 - P(y)]} / log y           (6.24)

and its sample analogue built from two order statistics (p. 200):

    m1 = n^(1 - delta1),  m2 = n^(1 - delta2),  0 < delta2 < delta1 < 1
    rho = 1 - (1/2) (n^-delta1 - n^-delta2) / ((delta1 - delta2) log n)
    a_n = -rho (delta1 - delta2) log n / (log Y_{m1:n} - log Y_{m2:n})   (6.25)
    sigma^2 = [1 / ((delta1 - delta2) log n)]^2 (n^delta1 - n^delta2)/n  (6.27)

The book adds, in the paragraph after (6.27), that with covariates alpha
is estimated by (6.25) applied as if the covariates were unobserved,
that gamma_n comes from the least-squares regression of log Y on X, and
that beta is then b_n = a_n gamma_n.  That is exactly what is done here.
The Weibull scale reported is exp(-beta_1), the baseline of (6.17) at
x = 0, which requires the first column of X to be the intercept.

Nothing is random: the estimator is a function of two order statistics
and an ordinary least-squares fit.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["horowitz_weibull_heterogeneity"]


def horowitz_weibull_heterogeneity(t, x, event=None, mixing_dist="nonparametric",
                                   delta1=0.6, delta2=0.3):
    """Honore (1990) estimator of the Weibull shape, then beta = alpha gamma.

    Parameters
    ----------
    t : array-like
        Durations Y, strictly positive.
    x : array-like
        n-by-p design matrix; the first column should be the intercept.
    event : array-like, optional
        1 for an observed duration, 0 to drop the observation.
    mixing_dist : str
        Label only.  The distribution of V is left unrestricted, which is
        the whole point of (6.24); anything other than "nonparametric" is
        refused rather than silently ignored.
    delta1, delta2 : float
        The exponents of p. 200, needing 0 < delta2 < delta1 < 1 and
        delta1 + 2 delta2 > 1 for the rate Honore proves.

    Returns
    -------
    alpha_hat : a_n of (6.25)
    beta_hat  : a_n gamma_n
    lambda_hat: exp(-beta_hat[0]), the baseline scale of (6.17)
    """
    tt = core.vec(t)
    XX = core.mat(x)
    n0 = len(tt)
    if n0 == 0:
        raise ValueError("horowitz_weibull_heterogeneity: t is empty")
    if len(XX) != n0:
        raise ValueError("horowitz_weibull_heterogeneity: x has a different number of rows than t")
    if mixing_dist != "nonparametric":
        raise ValueError("horowitz_weibull_heterogeneity: only the nonparametric mixing of (6.17) is offered")
    if event is not None:
        ev = core.vec(event)
        if len(ev) != n0:
            raise ValueError("horowitz_weibull_heterogeneity: event has a different length than t")
        keep = [i for i in range(n0) if ev[i] != 0.0]
    else:
        keep = list(range(n0))
    yv = [tt[i] for i in keep]
    Xk = [list(XX[i]) for i in keep]
    n = len(yv)
    if n < 3:
        raise ValueError("horowitz_weibull_heterogeneity: fewer than three uncensored durations")
    for v in yv:
        if v <= 0.0:
            raise ValueError("horowitz_weibull_heterogeneity: durations must be positive")
    d1 = float(delta1)
    d2 = float(delta2)
    if not (0.0 < d2 < d1 < 1.0):
        raise ValueError("horowitz_weibull_heterogeneity: need 0 < delta2 < delta1 < 1")
    srt = sorted(yv)
    ln = math.log(n)
    m1 = int(round(n ** (1.0 - d1)))
    m2 = int(round(n ** (1.0 - d2)))
    if m1 < 1:
        m1 = 1
    if m2 > n:
        m2 = n
    if m1 >= m2:
        raise ValueError("horowitz_weibull_heterogeneity: the two order statistics coincide, n is too small")
    rho = 1.0 - 0.5 * (n ** (-d1) - n ** (-d2)) / ((d1 - d2) * ln)
    den = math.log(srt[m1 - 1]) - math.log(srt[m2 - 1])
    if den == 0.0:
        raise ValueError("horowitz_weibull_heterogeneity: the two order statistics are tied")
    a_n = -rho * (d1 - d2) * ln / den
    s2 = (1.0 / ((d1 - d2) * ln)) ** 2 * (n ** d1 - n ** d2) / n
    ly = [math.log(v) for v in yv]
    gam = core.lstsq(Xk, ly, 1e-12)
    beta = [a_n * g for g in gam]
    lam = math.exp(-beta[0])
    return RichResult(
        title="Weibull hazard with unobserved heterogeneity (Honore 1990)",
        summary_lines=[("n", n), ("alpha", a_n)],
        payload={
            "estimate": a_n,
            "alpha_hat": a_n,
            "beta_hat": beta,
            "gamma_hat": gam,
            "lambda_hat": lam,
            "rho": rho,
            "m1": m1,
            "m2": m2,
            "sigma2": s2,
            "se_alpha": a_n * math.sqrt(s2),
            "n": n,
            "method": "Horowitz (2009) eq. (6.24)-(6.27) pp.199-200, Honore two-order-statistic alpha",
        },
    )


def cheatsheet():
    return "hrzweib: Weibull hazard model with unobserved heterogeneity"
