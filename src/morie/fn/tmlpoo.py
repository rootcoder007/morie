# morie.fn -- slice s03 (rootcoder007/morie)
"""Pooled TMLE across sites.

Source consulted: van der Laan, M. J. and Rubin, D. (2006), *The
International Journal of Biostatistics* 2(1), article 11, for the
targeting step, and Rothman, K. J., Greenland, S. and Lash, T. L.
(2008), *Modern Epidemiology*, 3rd ed., chapter 15, for the
fixed-effects pooling of site-specific estimates,

    psi_pool = sum_s w_s psi_s / sum_s w_s,   w_s = 1 / se_s^2

with variance 1 / sum_s w_s.  Heterogeneity is quantified by Cochran's Q
and Higgins and Thompson's I^2 (2002, *Statistics in Medicine* 21(11),
1539-1558),

    Q  = sum_s w_s (psi_s - psi_pool)^2
    I2 = max(0, (Q - (S - 1)) / Q)

Neither book chapter nor the 2002 paper was retrievable here as a full
text; the three expressions are quoted in their standard published form
and are not in dispute.  Fixed-effects pooling assumes a common effect;
I^2 is returned precisely so that assumption can be checked rather than
asserted.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["tmle_pooled"]


def tmle_pooled(y, D, X=None, site=None, alpha=0.05):
    """Site-stratified TMLE with inverse-variance pooling.

    Returns
    -------
    RichResult with payload:
        estimate : the pooled ATE
        se, ci_lo, ci_hi
        site_psi, site_se, site_n
        Q, I2, df
    """
    yv = k.vec(y)
    d = k.vec(D)
    n = len(yv)
    Xr = k.mat(X) if X is not None else None
    lab = [str(s) for s in (site if site is not None else [0] * n)]
    ids = []
    for s in lab:
        if s not in ids:
            ids.append(s)
    psis = []
    ses = []
    ns = []
    for s in ids:
        idx = [i for i in range(n) if lab[i] == s]
        ys = [yv[i] for i in idx]
        ds = [d[i] for i in idx]
        xs = [Xr[i] for i in idx] if Xr is not None else None
        ns.append(len(idx))
        if len(idx) < 3 or sum(ds) <= 0.0 or sum(ds) >= len(ds):
            psis.append(float("nan"))
            ses.append(float("nan"))
            continue
        f = k.tmle_ate(ys, ds, xs)
        psis.append(f["psi"])
        ses.append(f["se"])
    num = 0.0
    den = 0.0
    for i in range(len(ids)):
        if psis[i] == psis[i] and ses[i] == ses[i] and ses[i] > 0.0:
            w = 1.0 / (ses[i] * ses[i])
            num += w * psis[i]
            den += w
    pool = num / den if den > 0.0 else float("nan")
    sep = (1.0 / den) ** 0.5 if den > 0.0 else float("nan")
    Q = 0.0
    S = 0
    for i in range(len(ids)):
        if psis[i] == psis[i] and ses[i] == ses[i] and ses[i] > 0.0:
            w = 1.0 / (ses[i] * ses[i])
            Q += w * (psis[i] - pool) ** 2
            S += 1
    df = S - 1
    i2 = max(0.0, (Q - df) / Q) if Q > 0.0 else 0.0
    z = k.qnorm(1.0 - float(alpha) / 2.0)
    return RichResult(
        title="Pooled TMLE",
        summary_lines=[("pooled ATE", pool), ("sites", len(ids))],
        payload={
            "estimate": pool,
            "se": sep,
            "ci_lo": pool - z * sep,
            "ci_hi": pool + z * sep,
            "site_psi": psis,
            "site_se": ses,
            "site_n": ns,
            "Q": Q,
            "I2": i2,
            "df": df,
            "n": n,
            "method": "Site-stratified TMLE with inverse-variance (fixed-effects) pooling; Q and I^2 reported",
        },
    )


def cheatsheet():
    return "tmlpoo: Pooled TMLE for multi-site data"


tmlepooled = tmle_pooled
