# morie.fn -- function file (rootcoder007/morie)
"""Chernoff-Savage asymptotic normality of linear rank statistics."""

import math

from ._richresult import RichResult

__all__ = ['lrankasymp', 'gibbons_chernoff_savage']


def lrankasymp(j, jprime, lam, n, nodes=2001):
    """Null Chernoff-Savage mean and variance from the score function.

    Theorem 7.3.8 and Corollary 7.3.1 (book pp. 285-286).  Under H0,
    with lambda = m/N and J the limiting score-generating function,

    .. math:: \\mu = \\int_0^1 J(u)\\,du, \\qquad
        N\\lambda\\sigma^2 = 2(1-\\lambda)\\iint_{x<y}
            x(1-y)J'(x)J'(y)\\,dx\\,dy,

    and (T_N/m - mu)/sigma is asymptotically standard normal.  The
    double integral is reduced to one dimension by the inner
    cumulative A(y) = int_0^y x J'(x) dx, so the whole computation is
    a single fixed-grid pass (trapezoidal, ``nodes`` points) and is
    reproducible bit for bit.

    Parameters
    ----------
    j : callable
        The score function J on (0, 1).
    jprime : callable
        Its derivative J'.
    lam : float
        lambda = m/N, strictly inside (0, 1).
    n : int
        Total sample size N.
    nodes : int, optional
        Grid points on (0, 1) (default 2001).

    Returns
    -------
    RichResult
        keys ``mean``, ``var``, ``sd``, ``integral`` (the double
        integral), ``lam``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 7.3.8 and Corollary 7.3.1,
    pp. 285-286 (Chernoff and Savage, 1958).
    """
    lam = float(lam)
    n = int(n)
    nodes = int(nodes)
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must lie strictly inside (0, 1).")
    if n < 2:
        raise ValueError("n must be at least 2.")
    if nodes < 3:
        raise ValueError("nodes must be at least 3.")
    h = 1.0 / (nodes - 1)
    us = [k * h for k in range(nodes)]
    eps = 1e-9
    jv = [float(j(min(1.0 - eps, max(eps, u)))) for u in us]
    jp = [float(jprime(min(1.0 - eps, max(eps, u)))) for u in us]
    mu = 0.0
    for k in range(nodes - 1):
        mu += 0.5 * h * (jv[k] + jv[k + 1])
    acc = 0.0
    cum = [0.0] * nodes
    for k in range(1, nodes):
        acc += 0.5 * h * (us[k - 1] * jp[k - 1] + us[k] * jp[k])
        cum[k] = acc
    integ = 0.0
    for k in range(nodes - 1):
        f0 = (1.0 - us[k]) * jp[k] * cum[k]
        f1 = (1.0 - us[k + 1]) * jp[k + 1] * cum[k + 1]
        integ += 0.5 * h * (f0 + f1)
    var = 2.0 * (1.0 - lam) * integ / (n * lam)
    return RichResult(
        payload={
            "mean": float(mu),
            "var": float(var),
            "sd": float(math.sqrt(var)) if var > 0 else float("nan"),
            "integral": float(integ),
            "lam": lam,
            "n": n,
            "method": "Chernoff-Savage null moments (Thm 7.3.8, Cor 7.3.1)",
        }
    )


gibbons_chernoff_savage = lrankasymp
