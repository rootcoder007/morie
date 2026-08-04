# morie.fn -- function file (rootcoder007/morie)
"""Exact null distribution of the total number of runs (Table D)."""

import math

from ._richresult import RichResult

__all__ = ['runstab', 'gibbons_total_runs_dist_table']


def runstab(n1, n2, r=None):
    """Exact pmf and cdf of R over its whole support.

    Theorem 3.2.2 (book p. 79), eq. (3.2.3): under randomness

    .. math::
        P(R = 2k) = \\frac{2\\binom{n_1-1}{k-1}\\binom{n_2-1}{k-1}}
                          {\\binom{n}{n_1}}, \\quad
        P(R = 2k+1) = \\frac{\\binom{n_1-1}{k-1}\\binom{n_2-1}{k}
                          + \\binom{n_1-1}{k}\\binom{n_2-1}{k-1}}
                          {\\binom{n}{n_1}}.

    This is what Table D tabulates for small n1, n2; computing it is
    exact for any sizes rather than only those printed.

    Parameters
    ----------
    n1, n2 : int
        Counts of the two element types, each >= 1.
    r : int, optional
        Value at which to report the pmf and both tails.

    Returns
    -------
    RichResult
        keys ``support`` (2..n), ``pmf``, ``cdf``, ``pmf_r``,
        ``cdf_r``, ``sf_r``, ``mean``, ``var``, ``n1``, ``n2``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 3.2.2, eq. (3.2.3), p. 79;
    tabulated as Table D.
    """
    n1 = int(n1)
    n2 = int(n2)
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least 1.")
    n = n1 + n2
    den = math.comb(n, n1)
    support = list(range(2, n + 1))
    pmf = []
    for rr in support:
        if rr % 2 == 0:
            k = rr // 2
            p = 2.0 * math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k - 1)
        else:
            k = (rr - 1) // 2
            p = (
                math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k)
                + math.comb(n1 - 1, k) * math.comb(n2 - 1, k - 1)
            )
        pmf.append(p / den)
    cdf = []
    acc = 0.0
    for p in pmf:
        acc += p
        cdf.append(acc)
    mean = sum(s * p for s, p in zip(support, pmf))
    ex2 = sum(s * s * p for s, p in zip(support, pmf))
    out = {
        "support": support,
        "pmf": pmf,
        "cdf": cdf,
        "pmf_r": float("nan"),
        "cdf_r": float("nan"),
        "sf_r": float("nan"),
        "mean": float(mean),
        "var": float(ex2 - mean * mean),
        "n1": n1,
        "n2": n2,
        "method": "exact null distribution of R, eq. (3.2.3) (Table D)",
    }
    if r is not None:
        r = int(r)
        if not 2 <= r <= n:
            raise ValueError("r must lie in 2..n1+n2.")
        idx = r - 2
        out["pmf_r"] = pmf[idx]
        out["cdf_r"] = cdf[idx]
        out["sf_r"] = float(1.0 - (cdf[idx - 1] if idx > 0 else 0.0))
    return RichResult(payload=out)


gibbons_total_runs_dist_table = runstab
