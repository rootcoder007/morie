# morie.fn -- function file (rootcoder007/morie)
"""Sample size so that the EDF estimates F to within a fixed error."""

import math

from ._richresult import RichResult

__all__ = ['ksn', 'gibbons_ks_sample_size']


def ksn(c, alpha=0.05):
    """Smallest n with P(D_n < c) >= 1 - alpha.

    Section 4.4.3 (book p. 122): D_n bounds the error of S_n as a point
    estimate of F_X uniformly in x, so the minimum sample size that
    guarantees an error below c with probability 1 - alpha is the
    smallest n whose critical value D_{n,alpha} does not exceed c.  The
    asymptotic form is n = (k_alpha / c)^2 with k_alpha the Kolmogorov
    quantile; the exact answer is found by stepping n upward from that
    starting point using the exact cdf of Theorem 4.3.2.

    Parameters
    ----------
    c : float
        Tolerable uniform error, 0 < c < 1.
    alpha : float, optional
        Failure probability (default 0.05).

    Returns
    -------
    RichResult
        keys ``n``, ``n_asymp``, ``k_alpha``, ``coverage`` (attained
        P(D_n < c)), ``c``, ``alpha``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 4.4.3, p. 122.
    """
    from .gb432 import ksexact

    c = float(c)
    alpha = float(alpha)
    if not 0.0 < c < 1.0:
        raise ValueError("c must lie strictly inside (0, 1).")
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
    nasy = int(math.ceil((ka / c) ** 2))
    n = max(1, nasy - 20)
    cov = 0.0
    for _ in range(400):
        cov = ksexact(c, n)["cdf"]
        if cov >= 1.0 - alpha:
            break
        n += 1
    return RichResult(
        payload={
            "n": int(n),
            "n_asymp": int(nasy),
            "k_alpha": float(ka),
            "coverage": float(cov),
            "c": c,
            "alpha": alpha,
            "method": "KS sample size for uniform error c, Sec. 4.4.3",
        }
    )


gibbons_ks_sample_size = ksn
