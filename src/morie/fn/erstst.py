# morie.fn -- function file (rootcoder007/morie)
"""Elliott-Rothenberg-Stock GLS-detrended ADF."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["ers_unit_root"]


def ers_unit_root(x, lags=1, trend=False):
    """
    Elliott-Rothenberg-Stock GLS-detrended ADF (DF-GLS)

    Formula: local-to-unity GLS detrending at abar = 1 + cbar / T with
    cbar = -7 (mean case) or cbar = -13.5 (trend case).  Quasi-difference
    both the series and the deterministic terms,

        ytil_1 = y_1,  ytil_t = y_t - abar y_{t-1}
        ztil_1 = z_1,  ztil_t = z_t - abar z_{t-1}

    regress ytil on ztil to get psi, form the detrended series
    y^d = y - z psi, and run the ADF regression without deterministic
    terms

        dy^d_t = rho y^d_{t-1} + sum_{j=1}^{p} c_j dy^d_{t-j} + e_t.

    The DF-GLS statistic is the t-ratio on rho.

    Parameters
    ----------
    x : array-like
        Series.
    lags : int
        Number of lagged first differences p (>= 0).
    trend : bool
        Detrend against (1, t) rather than the constant alone.

    Returns
    -------
    result : dict
        Keys: estimate (the DF-GLS t-statistic), statistic, rho, se,
        abar, lags, nobs, n, method.

    References
    ----------
    Elliott, Rothenberg & Stock (1996), Econometrica 64(4):813-836,
    doi:10.2307/2171846.
    """
    y = [float(v) for v in x]
    n = len(y)
    p = int(lags)
    if p < 0:
        raise ValueError("lags must be non-negative")
    if n < p + 3:
        raise ValueError("series too short for %d lags" % p)
    cbar = -13.5 if trend else -7.0
    abar = 1.0 + cbar / n
    k = 2 if trend else 1
    Z = [[1.0] + ([float(t + 1)] if trend else []) for t in range(n)]
    yt = [y[0]] + [y[t] - abar * y[t - 1] for t in range(1, n)]
    Zt = [list(Z[0])] + [[Z[t][j] - abar * Z[t - 1][j] for j in range(k)]
                         for t in range(1, n)]
    psi = core.lstsq(Zt, yt, 0.0)
    yd = [y[t] - sum(Z[t][j] * psi[j] for j in range(k)) for t in range(n)]
    dy = [yd[t] - yd[t - 1] for t in range(1, n)]
    # dy is indexed 1..n-1 in original time; the ADF regression starts at
    # observation p+1 of dy so that all p lagged differences exist.
    rows = []
    rhs = []
    for i in range(p, len(dy)):
        row = [yd[i]]                     # y^d_{t-1}
        for j in range(1, p + 1):
            row.append(dy[i - j])
        rows.append(row)
        rhs.append(dy[i])
    nobs = len(rows)
    kk = p + 1
    if nobs <= kk:
        raise ValueError("series too short for %d lags" % p)
    b = core.lstsq(rows, rhs, 0.0)
    resid = [rhs[i] - sum(rows[i][j] * b[j] for j in range(kk))
             for i in range(nobs)]
    s2 = sum(r * r for r in resid) / (nobs - kk)
    XtX = [[sum(rows[i][a] * rows[i][c] for i in range(nobs))
            for c in range(kk)] for a in range(kk)]
    inv = core.cholsolve(XtX, [1.0 if j == 0 else 0.0 for j in range(kk)])
    se = math.sqrt(s2 * inv[0])
    stat = b[0] / se
    return RichResult(payload={
        "estimate": stat,
        "statistic": stat,
        "rho": b[0],
        "se": se,
        "abar": abar,
        "lags": p,
        "nobs": nobs,
        "n": n,
        "method": "Elliott-Rothenberg-Stock GLS-detrended ADF",
    })


def cheatsheet():
    return "erstst: Elliott-Rothenberg-Stock GLS-detrended ADF"


# compact alias per ledger/NAMING.md
ersunitroot = ers_unit_root
