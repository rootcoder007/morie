# morie.fn -- slice s05 (rootcoder007/morie)
"""BIC order selection for an autoregression, on Schwarz's own penalty.

Schwarz, G. (1978), "Estimating the dimension of a model", *The Annals
of Statistics* 6(2), 461-464, doi:10.1214/aos/1176344136.  The paper
was opened directly (Project Euclid) and its Proposition, page 462,
read off a rendered page image:

    S(Y, n, j) = n sup (Y . theta - b(theta)) - (1/2) k_j log n + R

with R bounded in n.  The Bayes solution maximises S, so on the
log-likelihood scale the penalty is (1/2) k log n and on the usual
deviance scale it is

    BIC = -2 log L_max + k log n.

That factor is the entire content of the criterion and the entire
difference from AIC: the penalty per parameter GROWS with the sample,
log n against 2, so BIC is consistent for the true order where AIC is
not, and for any n > 7 it is the stricter of the two.

For an autoregression of order p fitted by conditional least squares
with an intercept, k = p + 2 -- intercept, p coefficients and the
innovation variance.  Every candidate order is fitted on the SAME
T = n - max_p observations, so the likelihoods being compared are
likelihoods of the same data; refitting each order on as many
observations as it can use makes the criteria incomparable, which is
the standard way this selection is got wrong.

Two scalings are returned.  ``bic_raw`` is Schwarz's own
-2 log L + k log T.  ``bic`` is the per-observation form quoted in the
time-series literature,

    BIC(p) = log(sigma_p^2) + p log(T) / T,

which differs from bic_raw / T by log(2 pi) + 1 + 2 log(T)/T -- a
constant in p -- and therefore selects the same order.  The two are
computed independently here and their agreement on the argmin is a
check, not an assumption.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["bic_ar_order"]


def bic_ar_order(x, max_p):
    """Select an autoregressive order by the Schwarz criterion.

    Parameters
    ----------
    x : array-like
        The series, in time order.
    max_p : int
        Largest order considered; orders 0, 1, ..., max_p are compared.

    Returns
    -------
    RichResult
        keys: ``estimate`` (the selected order), ``order``, ``bic``
        (per-observation criterion, one entry per order), ``bic_raw``
        (-2 log L + k log T), ``sigma2``, ``coefficients`` (intercept
        first, for the selected order), ``n``, ``T``, ``max_p``,
        ``method``.

    References
    ----------
    Schwarz, G. (1978), *Annals of Statistics* 6(2):461-464,
    doi:10.1214/aos/1176344136, Proposition, p. 462.
    """
    xv = core.vec(x)
    n = len(xv)
    P = int(max_p)
    if P != max_p or P < 0:
        raise ValueError("bic_ar_order: max_p must be a non-negative integer")
    T = n - P
    if T < P + 3:
        raise ValueError(
            "bic_ar_order: too few observations; %d points leave T = %d for "
            "order %d, which cannot support %d parameters"
            % (n, T, P, P + 2))
    y = xv[P:]
    bic = []
    bic_raw = []
    sig2 = []
    coefs = []
    logT = math.log(T)
    for p in range(P + 1):
        X = []
        for t in range(P, n):
            row = [1.0]
            for lag in range(1, p + 1):
                row.append(xv[t - lag])
            X.append(row)
        beta = core.lstsq(X, y, ridge=0.0)
        rss = 0.0
        for i in range(T):
            fit = 0.0
            for j in range(p + 1):
                fit += X[i][j] * beta[j]
            rss += (y[i] - fit) ** 2
        s2 = rss / T
        if not s2 > 0.0:
            raise ValueError(
                "bic_ar_order: the order-%d fit is exact, so the Gaussian "
                "likelihood is unbounded and no BIC exists" % p)
        sig2.append(s2)
        coefs.append([float(b) for b in beta])
        bic.append(math.log(s2) + p * logT / T)
        # Schwarz's own scale: k = p + 2 (intercept, p lags, sigma^2)
        bic_raw.append(T * (math.log(2.0 * math.pi * s2) + 1.0)
                       + (p + 2) * logT)
    best = min(range(P + 1), key=lambda p: bic[p])
    best_raw = min(range(P + 1), key=lambda p: bic_raw[p])
    if best != best_raw:
        raise ValueError(
            "bic_ar_order: the two scalings of the criterion disagree on the "
            "argmin, which is arithmetically impossible; the fit is degenerate")
    return RichResult(payload={
        "estimate": int(best), "order": int(best),
        "bic": bic, "bic_raw": bic_raw, "sigma2": sig2,
        "coefficients": coefs[best],
        "n": int(n), "T": int(T), "max_p": int(P),
        "method": "Schwarz (1978) BIC = -2 log L + k log n, AR order selection"})


def cheatsheet():
    return ("bicarp: penalty log n per parameter, not 2 -- BIC is consistent "
            "where AIC is not, and every order must be fitted on the same T")


# compact alias per ledger/NAMING.md
bicarorder = bic_ar_order
