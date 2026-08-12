"""Copula-based mutual information (Linfoot 1957; Calsaverini & Vicente 2009)."""

import math

from ._stats_core import norm
from ._richresult import RichResult

__all__ = ["cmuti", "copula_mutual_information"]


def _ranks_to_unit(x):
    # empirical copula transform: rank / (n + 1), average ranks on ties
    n = len(x)
    idx = sorted(range(n), key=lambda i: x[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[idx[t]] = avg
        i = j + 1
    return [v / (n + 1.0) for v in r]


def cmuti(x, y):
    """
    Mutual information of a bivariate sample via its Gaussian copula.

    The mutual information of (X, Y) depends only on the copula, not
    on the marginals (Calsaverini & Vicente 2009: MI is the
    dependence component of the information content and is
    marginal-invariant).  This estimator therefore (i) maps each
    variable to the unit square by the empirical copula transform
    rank/(n+1), (ii) applies normal scores z = Phi^-1(u), and (iii)
    evaluates the Gaussian-copula closed form printed as the paper's
    minimal MI of a Gaussian copula,

        I_Gauss(rho) = -(1/2) log(1 - rho^2),

    at the normal-scores correlation rho.  Also reported is
    Linfoot's (1957) informational coefficient of correlation
    r_1 = sqrt(1 - exp(-2 I)), which equals |rho| exactly in the
    Gaussian case.

    Sources
    -------
    Calsaverini, R. S. & Vicente, R. (2009). An information-theoretic
    approach to statistical dependence: Copula information. *EPL*,
    88, 68003, Eq. for I_Gauss and the marginal-invariance discussion
    (local copy
    fetched-wave3/calsaverini-vicente-2009-copula-information-epl88.pdf).
    Linfoot, E. H. (1957). An informational measure of correlation.
    *Information and Control*, 1, 85-89 (r_1 = sqrt(1 - e^{-2I})).

    Parameters
    ----------
    x, y : sequences of float
        Paired observations (n >= 3).

    Returns
    -------
    RichResult
        Keys: estimate (MI in nats), rho (normal-scores
        correlation), linfoot_r, n.
    """
    xv = [float(v) for v in x]
    yv = [float(v) for v in y]
    n = len(xv)
    if len(yv) != n or n < 3:
        raise ValueError("x and y must be paired with n >= 3")
    u = _ranks_to_unit(xv)
    v = _ranks_to_unit(yv)
    zx = [float(norm.ppf(t)) for t in u]
    zy = [float(norm.ppf(t)) for t in v]
    mx = sum(zx) / n
    my = sum(zy) / n
    sxx = sum((a - mx) ** 2 for a in zx)
    syy = sum((a - my) ** 2 for a in zy)
    sxy = sum((a - mx) * (b - my) for a, b in zip(zx, zy))
    if sxx <= 0 or syy <= 0:
        raise ValueError("degenerate sample (a variable is constant)")
    rho = sxy / math.sqrt(sxx * syy)
    rho = max(min(rho, 1.0 - 1e-12), -1.0 + 1e-12)
    mi = -0.5 * math.log(1.0 - rho * rho)
    return RichResult(payload={
        "estimate": mi,
        "rho": rho,
        "linfoot_r": math.sqrt(1.0 - math.exp(-2.0 * mi)),
        "n": n,
        "method": "Gaussian-copula MI, I = -log(1-rho^2)/2 "
                  "(Calsaverini & Vicente 2009; Linfoot 1957)",
    })


# long descriptive alias (stub-era name)
copula_mutual_information = cmuti


def cheatsheet():
    return "cmuti: normal-scores rho -> MI = -log(1-rho^2)/2, marginal-invariant"
