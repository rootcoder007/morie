# morie.fn -- function file (rootcoder007/morie)
"""Power function of the sign test -- exact and normal approximation."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['signpow', 'gibbons_sign_power']


def signpow(n, theta, alpha=0.05, exact=True):
    """Power of the upper-tailed sign test against H1: M > M0.

    Book p. 173-174.  The exact power is the binomial tail

    .. math:: Pw(\\theta) = \\sum_{i=k_\\alpha}^{N}
        \\binom{N}{i}\\theta^i (1-\\theta)^{N-i},

    with k_alpha the smallest integer whose null upper tail is at most
    alpha.  The normal approximation is eq. (5.4.8),

    .. math:: Pw = 1 - \\Phi\\!\\left[
        \\frac{N(0.5-\\theta) + 0.5\\sqrt{N} z_\\alpha}
             {\\sqrt{N\\theta(1-\\theta)}}\\right].

    Parameters
    ----------
    n : int
        Sample size.
    theta : float
        P(X > M0) under the alternative, 0 < theta < 1.
    alpha : float, optional
        Nominal size (default 0.05).
    exact : bool, optional
        Also compute the exact binomial power (default True).

    Returns
    -------
    RichResult
        keys ``power`` (normal approximation, eq. 5.4.8),
        ``power_exact``, ``k_alpha``, ``alpha_exact``, ``n``,
        ``theta``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (5.4.8), Table 5.4.1, p. 174.
    """
    n = int(n)
    theta = float(theta)
    alpha = float(alpha)
    if n < 1:
        raise ValueError("n must be at least 1.")
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must lie strictly inside (0, 1).")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")
    za = stats.norm.ppf(1.0 - alpha)
    approx = 1.0 - stats.norm.cdf(
        (n * (0.5 - theta) + 0.5 * math.sqrt(n) * za)
        / math.sqrt(n * theta * (1.0 - theta))
    )
    ka = n
    aex = float("nan")
    pex = float("nan")
    if exact:
        half = 0.5**n
        for c in range(n + 1):
            tail = sum(math.comb(n, i) for i in range(c, n + 1)) * half
            if tail <= alpha:
                ka = c
                aex = tail
                break
        pex = sum(
            math.comb(n, i) * theta**i * (1.0 - theta) ** (n - i)
            for i in range(ka, n + 1)
        )
    return RichResult(
        payload={
            "power": float(approx),
            "power_exact": float(pex),
            "k_alpha": int(ka),
            "alpha_exact": float(aex),
            "n": n,
            "theta": theta,
            "method": "sign test power, eq. (5.4.8) with exact binomial tail",
        }
    )


gibbons_sign_power = signpow
