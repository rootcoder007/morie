# morie.fn -- function file (rootcoder007/morie)
"""Critical values of the one-sample KS statistic (Table F)."""

import math

from ._richresult import RichResult

__all__ = ['kscrit', 'gibbons_ks_critical_values']


def kscrit(n, alpha=0.05, exact=True):
    """Two-sided critical value D_{n,alpha} of the KS statistic.

    Table F (book p. 565) tabulates the values solving
    P(D_n >= D_{n,alpha}) = alpha.  Rather than transcribing the table,
    the exact value is obtained by bisecting the exact cdf of Theorem
    4.3.2 (``morie.fn.gb432.ksexact``) over 200 fixed steps -- a fixed
    iteration count, so the answer is bit-identical across languages.
    The asymptotic value is the Kolmogorov limit, k_alpha/sqrt(n) with
    Q(k) = 2 sum_j (-1)^{j-1} exp(-2 j^2 k^2) = alpha.

    Parameters
    ----------
    n : int
        Sample size.
    alpha : float, optional
        Upper-tail probability (default 0.05).
    exact : bool, optional
        Bisect the exact cdf (default True); otherwise only the
        asymptotic value is returned.

    Returns
    -------
    RichResult
        keys ``dcrit``, ``dcrit_asymp``, ``k_alpha``, ``n``,
        ``alpha``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Table F, p. 565; Theorem 4.3.3
    (Kolmogorov limit), p. 108.
    """
    from .gb432 import ksexact

    n = int(n)
    alpha = float(alpha)
    if n < 1:
        raise ValueError("n must be at least 1.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")

    def q(k):
        s = 0.0
        for j in range(1, 101):
            s += (-1.0) ** (j - 1) * math.exp(-2.0 * j * j * k * k)
        return 2.0 * s

    lo, hi = 1e-6, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if q(mid) > alpha:
            lo = mid
        else:
            hi = mid
    ka = (lo + hi) / 2.0
    dasy = ka / math.sqrt(n)
    dex = float("nan")
    if exact:
        lo, hi = 1e-9, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if 1.0 - ksexact(mid, n)["cdf"] > alpha:
                lo = mid
            else:
                hi = mid
        dex = (lo + hi) / 2.0
    return RichResult(
        payload={
            "dcrit": float(dex),
            "dcrit_asymp": float(dasy),
            "k_alpha": float(ka),
            "n": n,
            "alpha": alpha,
            "method": "KS critical value (Table F) by exact bisection",
        }
    )


gibbons_ks_critical_values = kscrit
