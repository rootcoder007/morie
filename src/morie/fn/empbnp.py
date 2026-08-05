# morie.fn -- function file (rootcoder007/morie)
"""Robbins' nonparametric empirical Bayes rule."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["empirical_bayes_np"]


def empirical_bayes_np(y, prior_family="poisson"):
    """
    Robbins' nonparametric empirical Bayes

    Formula: E[theta | y = k] = (k + 1) f(k + 1) / f(k)

    For Poisson observations the posterior mean of the rate needs no
    parametric prior at all: it is recovered from the marginal counts
    alone.  The gaussian branch is Tweedie's formula
    E[theta | y] = y + d/dy log f(y), estimated from a kernel-smoothed
    marginal.

    Parameters
    ----------
    y : array-like
        Observed counts (poisson) or values (gaussian).
    prior_family : str
        "poisson" for Robbins' rule, "gaussian" for Tweedie's.

    Returns
    -------
    result : dict
        Keys: estimate (mean of the fitted posterior means), theta_hat,
        support, counts, n.

    References
    ----------
    Robbins (1956), Proc. 3rd Berkeley Symp. 1:157-163.
    Efron (2011), JASA 106(496):1602-1614 (Tweedie's formula).
    """
    y = core.vec(y)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    fam = str(prior_family).lower()
    if fam not in ("poisson", "gaussian"):
        raise ValueError("prior_family must be 'poisson' or 'gaussian'")
    if fam == "poisson":
        ks = [int(round(v)) for v in y]
        if any(v < 0 for v in ks):
            raise ValueError("poisson counts must be non-negative")
        top = max(ks)
        cnt = [0] * (top + 2)
        for v in ks:
            cnt[v] += 1
        support = list(range(top + 1))
        theta = []
        for k in support:
            theta.append((k + 1.0) * cnt[k + 1] / cnt[k] if cnt[k] > 0
                         else float("nan"))
        per = [theta[v] for v in ks]
        est = sum(v for v in per if v == v) / max(
            sum(1 for v in per if v == v), 1)
        return RichResult(payload={
            "estimate": est,
            "theta_hat": theta,
            "support": support,
            "counts": cnt[:top + 1],
            "n": n,
            "method": "Robbins nonparametric empirical Bayes (Poisson)",
        })
    # Tweedie: a Gaussian kernel estimate of log f and its derivative
    s = core.sd(y, 1)
    if s <= 0.0:
        raise ValueError("y has zero spread; Tweedie's formula is undefined")
    h = 1.06 * s * n ** (-0.2)
    theta = []
    for i in range(n):
        f = 0.0
        fp = 0.0
        for j in range(n):
            u = (y[i] - y[j]) / h
            k = math.exp(-0.5 * u * u)
            f += k
            fp += k * (-u / h)
        theta.append(y[i] + fp / f if f > 0.0 else y[i])
    return RichResult(payload={
        "estimate": sum(theta) / n,
        "theta_hat": theta,
        "support": list(y),
        "counts": [1] * n,
        "n": n,
        "method": "Tweedie nonparametric empirical Bayes (Gaussian)",
    })


def cheatsheet():
    return "empbnp: Robbins nonparametric empirical Bayes"


# compact alias per ledger/NAMING.md
empiricalbayesnp = empirical_bayes_np
