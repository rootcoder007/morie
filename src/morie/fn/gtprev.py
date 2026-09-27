"""Prevalence from group (pooled) testing: MLE, intervals and empirical Bayes estimates.

Bilder & Loughin (2025), Analysis of Categorical Data with R, eqs (6.30)-(6.31), Sec 6.4.
"""

import math

from ._richresult import RichResult
from ._rrng_core import qbeta, qnorm

__all__ = ["gtprev"]


def _fmin(f, ax, bx, tol):
    """Brent's localmin as in R's optimize (Brent_fmin)."""
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(2.220446049250313e-16)
    a, b = ax, bx
    v = w = x = a + c * (b - a)
    d = e = 0.0
    fx = fv = fw = f(x)
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v in (x, w):
                v, fv = u, fu
    return x


def gtprev(x, m, n, ci="CP", alpha=0.05, b_range=(0.0001, 1000.0)):
    r"""Prevalence pi from x positive groups among n groups of m specimens each.

    theta = P(group positive) = 1 - (1 - pi)^m (6.30), so the MLE is
    pi_hat = 1 - (1 - x/n)^{1/m} (6.31). Intervals: ``"CP"`` and ``"score"``
    transform the Clopper-Pearson or Wilson interval for theta through the same
    map; ``"Wald"`` uses the delta method,
    se(pi_hat) = sqrt(theta(1-theta)/n) (1-theta)^{1/m - 1}/m, truncated at 0 and 1.
    The empirical Bayes estimators of Bilder & Tebbs take b_hat maximising
    g(b) = log b + lgamma(n - x + b/m) - lgamma(n + b/m + 1) (Brent's method over
    ``b_range``) and give EB1 = 1 - exp(lgamma(n - x + b/m + 1/m) + lgamma(n + b/m + 1)
    - lgamma(n + b/m + 1 + 1/m) - lgamma(n - x + b/m)) and
    EB2 = 1 - (1 - (x + 1)/(n + b/m + 1))^{1/m}.

    Parameters
    ----------
    x : int
        Positive groups.
    m : int
        Group size.
    n : int
        Number of groups.
    ci : {"CP", "Wald", "score"}
    alpha : float
    b_range : (float, float)

    Returns
    -------
    RichResult
        Keys: estimate, ci, theta, b_hat, eb1, eb2.

    References
    ----------
    Bilder, C. R. & Tebbs, J. M. (2009). Statistics in Medicine 28, 2340-2355.
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Eqs (6.30)-(6.31).

    Examples
    --------
    >>> round(gtprev(3, 7, 24)["estimate"], 12)
    0.018895119427
    """
    if not (0 <= x <= n and m >= 1 and n >= 1):
        raise ValueError("need 0 <= x <= n, m >= 1 and n >= 1")
    th = x / n
    tr = lambda t: 1 - (1 - t) ** (1 / m)  # noqa: E731
    est = tr(th)
    z = qnorm(1 - alpha / 2)
    if ci == "CP":
        lo = 0.0 if x == 0 else qbeta(alpha / 2, x, n - x + 1)
        hi = 1.0 if x == n else qbeta(1 - alpha / 2, x + 1, n - x)
        interval = (tr(lo), tr(hi))
    elif ci == "score":
        cen = (x + z * z / 2) / (n + z * z)
        half = z * math.sqrt(x * (n - x) / n + z * z / 4) / (n + z * z)
        interval = (tr(max(0.0, cen - half)), tr(min(1.0, cen + half)))
    elif ci == "Wald":
        s = math.sqrt(th * (1 - th) / n) * (1 - th) ** (1 / m - 1) / m if 0 < th < 1 else 0.0
        interval = (max(0.0, est - z * s), min(1.0, est + z * s))
    else:
        raise ValueError('ci must be "CP", "Wald" or "score"')
    K, t, gs = n, x, m

    def negg(b):
        return -(math.log(b) + math.lgamma(K - t + b / gs) - math.lgamma(K + b / gs + 1))

    b = _fmin(negg, b_range[0], b_range[1], 2.220446049250313e-16**0.25)
    eb1 = 1 - math.exp(
        math.lgamma(K - t + b / gs + 1 / gs)
        + math.lgamma(K + b / gs + 1)
        - math.lgamma(K + b / gs + 1 + 1 / gs)
        - math.lgamma(K - t + b / gs)
    )
    eb2 = 1 - (1 - (t + 1) / (K + b / gs + 1)) ** (1 / gs)
    return RichResult(
        title="Group testing prevalence",
        summary_lines=[("estimate", est), ("ci", interval)],
        payload={"estimate": est, "ci": interval, "theta": th, "b_hat": b, "eb1": eb1, "eb2": eb2},
    )


def cheatsheet():
    return "gtprev: pooled-testing prevalence 1-(1-x/n)^(1/m), CP/score/Wald CIs, EB estimates. Bilder & Loughin eqs (6.30)-(6.31)."
