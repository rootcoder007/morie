# morie.fn -- slice k04 (rootcoder007/morie)
"""Rosner (1983) generalized extreme Studentized deviate (ESD) test.

Source FETCHED: NIST/SEMATECH e-Handbook of Statistical Methods,
section 1.3.5.17.3 "Generalized ESD Test for Outliers", which states
Rosner (1983, *Technometrics* 25, 165-172) in full:

    R_i = max_i |x_i - xbar| / s      (recomputed after each removal)

    lambda_i = (n - i) t_{p, n-i-1}
               / sqrt( (n - i - 1 + t_{p,n-i-1}^2) (n - i + 1) )

    p = 1 - alpha / (2 (n - i + 1)),    i = 1, ..., r

The number of outliers is the largest ``i`` with ``R_i > lambda_i``.
The handbook worked example (Rosner own 54-point data set, r = 10,
alpha = 0.05) gives R_1 = 3.118, lambda_1 = 3.158 and three outliers;
that example is the regression test for this function.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["generalized_esd"]


def generalized_esd(x, alpha=0.05, r=None):
    """Generalized ESD test for up to ``r`` outliers.

    Parameters
    ----------
    x : array-like
        Univariate sample, approximately normal under H0.
    alpha : float, default 0.05
        Significance level.
    r : int, optional
        Upper bound on the number of outliers.  Defaults to
        ``max(1, n // 10)``.

    Returns
    -------
    RichResult
        keys: ``n_outliers``, ``outlier_index`` (indices into ``x``),
        ``R`` (the r test statistics), ``lam`` (the r critical values),
        ``alpha``, ``r``, ``n``, ``method``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if r is None:
        r = max(1, n // 10)
    r = int(r)
    if n < 3 or r < 1 or r > n - 2:
        raise ValueError("need n>=3 and 1<=r<=n-2")
    alpha = float(alpha)

    keep = list(range(n))
    R = []
    removed = []
    for _ in range(r):
        vals = x[np.array(keep)]
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        dev = np.abs(vals - m)
        j = int(np.argmax(dev))
        R.append(float(dev[j] / s) if s > 0.0 else float("inf"))
        removed.append(keep[j])
        keep.pop(j)

    lam = []
    for i in range(1, r + 1):
        p = 1.0 - alpha / (2.0 * (n - i + 1))
        nu = n - i - 1
        t = float(stats.t.ppf(p, nu))
        lam.append((n - i) * t / np.sqrt((nu + t * t) * (n - i + 1)))

    n_out = 0
    for i in range(r):
        if R[i] > lam[i]:
            n_out = i + 1
    return RichResult(
        payload={
            "n_outliers": n_out,
            "outlier_index": np.array(removed[:n_out], dtype=int),
            "R": np.array(R, dtype=float),
            "lam": np.array(lam, dtype=float),
            "alpha": alpha,
            "r": r,
            "n": n,
            "method": "Generalized ESD test (Rosner 1983)",
        }
    )


def cheatsheet():
    return "esd: generalized ESD outlier test (Rosner 1983)"


# compact alias per ledger/NAMING.md
generalizedesd = generalized_esd
