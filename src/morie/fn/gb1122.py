# morie.fn -- function file (rootcoder007/morie)
"""Exact and asymptotic null distribution of Kendall's T."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['taunull', 'gibbons_kendall_null']


def taunull(n, s=None):
    """Null law of the concordance excess S = P - Q for Kendall's tau.

    Section 11.2.1 (book p. 395).  Under independence every one of the
    n! rankings is equally likely; the number of inversions follows the
    classical Mahonian recursion, so the distribution of
    S = P - Q = n(n-1)/2 - 2 (inversions) is obtained exactly by
    dynamic programming over the inversion generating function

    .. math:: \\prod_{i=1}^{n}(1 + z + \\dots + z^{i-1}).

    S is symmetric about 0 and its asymptotic variance is that of tau
    scaled: Var[T] = 2(2n+5)/[9n(n-1)].

    Parameters
    ----------
    n : int
        Number of pairs, n >= 2.
    s : int, optional
        Value of S at which to report the pmf and both tails.

    Returns
    -------
    RichResult
        keys ``support`` (values of S), ``pmf``, ``pmf_s``, ``cdf_s``,
        ``sf_s``, ``var_tau``, ``var_s``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 11.2.1, p. 395; Table L, p. 583.
    """
    n = int(n)
    if n < 2:
        raise ValueError("n must be at least 2.")
    maxinv = n * (n - 1) // 2
    counts = [0.0] * (maxinv + 1)
    counts[0] = 1.0
    for i in range(2, n + 1):
        new = [0.0] * (maxinv + 1)
        run = 0.0
        for k in range(maxinv + 1):
            run += counts[k]
            if k - i >= 0:
                run -= counts[k - i]
            new[k] = run
        counts = new
    tot = sum(counts)
    pmf = [c / tot for c in counts]
    support = [maxinv - 2 * k for k in range(maxinv + 1)]
    npairs = maxinv
    var_tau = 2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0))
    out = {
        "support": support,
        "pmf": pmf,
        "pmf_s": float("nan"),
        "cdf_s": float("nan"),
        "sf_s": float("nan"),
        "var_tau": float(var_tau),
        "var_s": float(var_tau * npairs * npairs),
        "n": n,
        "method": "null distribution of S = P - Q (Sec. 11.2.1)",
    }
    if s is not None:
        sv = int(s)
        if (maxinv - sv) % 2 != 0 or not -maxinv <= sv <= maxinv:
            raise ValueError("s is outside the support of S.")
        idx = (maxinv - sv) // 2
        out["pmf_s"] = pmf[idx]
        out["cdf_s"] = float(sum(pmf[idx:]))
        out["sf_s"] = float(sum(pmf[: idx + 1]))
    return RichResult(payload=out)


gibbons_kendall_null = taunull
