# morie.fn -- function file (rootcoder007/morie)
"""Exact null distribution of the two-sided KS statistic D_n."""

import math

from ._richresult import RichResult

__all__ = ['ksexact', 'gibbons_ks_exact_dist']


def ksexact(d, n):
    """P(D_n < d) by the Durbin matrix form of Theorem 4.3.2.

    Theorem 4.3.2 (book p. 111) writes P(D_n < 1/(2n) + v) as a nested
    integral of the uniform order-statistic density over the band
    |u_i - (2i-1)/(2n)| < v.  Evaluating that integral in closed form
    gives Durbin's (1973) matrix identity

    .. math:: P(D_n < d) = \\frac{n!}{n^n} (H^n)_{kk},

    with k = ceil(nd), t = k - nd and H the (2k-1) x (2k-1) matrix of
    the standard construction.  This routine computes exactly that, so
    it agrees with the theorem to machine precision without numerical
    quadrature.

    Parameters
    ----------
    d : float
        Value of the statistic, 0 < d < 1.
    n : int
        Sample size, n >= 1.

    Returns
    -------
    RichResult
        keys ``cdf`` (P(D_n < d)), ``sf``, ``k``, ``t``, ``n``, ``d``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 4.3.2, p. 111; evaluated by
    Durbin's matrix identity.
    """
    d = float(d)
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1.")
    if d <= 0.0:
        return RichResult(
            payload={"cdf": 0.0, "sf": 1.0, "k": 0, "t": 0.0, "n": n,
                     "d": d, "method": "P(D_n < d), Durbin matrix form"}
        )
    if d >= 1.0:
        return RichResult(
            payload={"cdf": 1.0, "sf": 0.0, "k": n, "t": 0.0, "n": n,
                     "d": d, "method": "P(D_n < d), Durbin matrix form"}
        )
    k = int(math.ceil(n * d))
    t = k - n * d
    m = 2 * k - 1
    h = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i - j + 1 >= 0:
                h[i][j] = 1.0
    for i in range(m):
        h[i][0] -= t ** (i + 1)
        h[m - 1][i] -= t ** (m - i)
    h[m - 1][0] += (2.0 * t - 1.0) ** m if 2.0 * t - 1.0 > 0.0 else 0.0
    for i in range(m):
        for j in range(m):
            if i - j + 1 > 0:
                for g in range(1, i - j + 2):
                    h[i][j] /= g
    # H^n by repeated squaring, tracking the exponent scale
    eq = 0
    res = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    base = [row[:] for row in h]
    e_base = 0
    p = n
    while p > 0:
        if p & 1:
            res, eq = _matmul(res, base, m), eq + e_base
            res, eq = _rescale(res, eq, m)
        p >>= 1
        if p:
            base = _matmul(base, base, m)
            e_base = 2 * e_base
            base, e_base = _rescale(base, e_base, m)
    s = res[k - 1][k - 1]
    for i in range(1, n + 1):
        s = s * i / n
        if s < 1e-140:
            s *= 1e140
            eq += 140
    val = s * 10.0 ** (-eq) if eq else s
    val = min(1.0, max(0.0, val))
    return RichResult(
        payload={
            "cdf": float(val),
            "sf": float(1.0 - val),
            "k": int(k),
            "t": float(t),
            "n": n,
            "d": d,
            "method": "P(D_n < d), exact (Thm 4.3.2 via Durbin matrix)",
        }
    )


def _matmul(a, b, m):
    out = [[0.0] * m for _ in range(m)]
    for i in range(m):
        ai = a[i]
        oi = out[i]
        for kk in range(m):
            aik = ai[kk]
            if aik == 0.0:
                continue
            bk = b[kk]
            for j in range(m):
                oi[j] += aik * bk[j]
    return out


def _rescale(a, e, m):
    v = a[m // 2][m // 2]
    if v > 1e140:
        for i in range(m):
            for j in range(m):
                a[i][j] *= 1e-140
        e += 140
    return a, e


gibbons_ks_exact_dist = ksexact
