# morie.fn -- function file (rootcoder007/morie)
"""Sign test for the median: the count of positive differences."""

import math

from ._richresult import RichResult

__all__ = ['signk', 'gibbons_sign_test']


def signk(x, m0=0.0):
    """Sign-test statistic K = #{X_i > M0} and its null moments.

    Section 5.4 (book p. 168), eq. (5.4.1): under H0: M = M0 the count
    of positive differences K is Binomial(N, 1/2), so E[K] = N/2 and
    Var[K] = N/4.  Zero differences are dropped and N reduced, the
    convention of Sec. 5.4.8.

    Parameters
    ----------
    x : sequence of float
        Sample (or paired differences).
    m0 : float, optional
        Hypothesised median (default 0).

    Returns
    -------
    RichResult
        keys ``statistic`` (K), ``n`` (non-zero differences), ``nzero``,
        ``mean``, ``var``, ``n_raw``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.4, eq. (5.4.1), p. 168.
    """
    xs = [float(v) - float(m0) for v in x]
    n_raw = len(xs)
    if n_raw < 1:
        raise ValueError("x must be non-empty.")
    nz = sum(1 for d in xs if d == 0.0)
    n = n_raw - nz
    k = sum(1 for d in xs if d > 0.0)
    return RichResult(
        payload={
            "statistic": int(k),
            "n": int(n),
            "nzero": int(nz),
            "mean": n / 2.0,
            "var": n / 4.0,
            "n_raw": int(n_raw),
            "method": "sign test K = #{X_i > M0} ~ Bin(N, 1/2)",
        }
    )


gibbons_sign_test = signk
