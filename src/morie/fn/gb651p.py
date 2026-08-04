# morie.fn -- function file (rootcoder007/morie)
"""Power of the control median test under a specified alternative."""

import math

from ._richresult import RichResult

__all__ = ['ctrlmedpow', 'gibbons_ctrl_median_power']


def ctrlmedpow(m, n, d, h, nodes=2001):
    """P(V <= d) under an alternative, Sec. 6.5.2.

    Book p. 258-259.  The alternative law of V is the mixture obtained
    by conditioning on the control median: with h = F_X o F_Y^{-1},

    .. math:: P[V = j] = \\binom{m}{j}\\frac{n!}{r!\\,r!}
        \\int_0^1 h(v)^j [1-h(v)]^{m-j} v^r (1-v)^r\\,dv,

    and the power of the lower-tailed test is the sum for j <= d.
    Setting h(v) = v reproduces eq. (6.5.1) exactly, which is the
    built-in check on this routine.  The integral uses composite
    Simpson on a fixed grid, so both language arms agree bit for bit.

    Parameters
    ----------
    m : int
        Size of the X sample.
    n : int
        Size of the Y (control) sample, odd.
    d : int
        Rejection region is V <= d.
    h : callable
        v -> F_X(F_Y^{-1}(v)) on [0, 1]; ``lambda v: v`` gives H0.
    nodes : int, optional
        Simpson nodes (default 2001, forced odd).

    Returns
    -------
    RichResult
        keys ``power``, ``pmf``, ``q`` (the implied F_X(M_Y) = h(0.5)),
        ``r``, ``m``, ``n``, ``d``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.5.2, p. 258.
    """
    from .gb641p import _simpson

    m = int(m)
    n = int(n)
    d = int(d)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    if n % 2 == 0:
        raise ValueError("the control sample size n must be odd (n = 2r+1).")
    r = (n - 1) // 2
    coef = math.factorial(n) / (math.factorial(r) * math.factorial(r))
    pmf = []
    for j in range(m + 1):
        def integrand(v, j=j):
            hv = float(h(v))
            hv = min(1.0, max(0.0, hv))
            return hv**j * (1.0 - hv) ** (m - j) * v**r * (1.0 - v) ** r

        pmf.append(math.comb(m, j) * coef * _simpson(integrand, nodes))
    power = sum(pmf[: min(d + 1, m + 1)]) if d >= 0 else 0.0
    return RichResult(
        payload={
            "power": float(power),
            "pmf": pmf,
            "q": float(h(0.5)),
            "r": int(r),
            "m": m,
            "n": n,
            "d": d,
            "method": "control median test power, Sec. 6.5.2",
        }
    )


gibbons_ctrl_median_power = ctrlmedpow
