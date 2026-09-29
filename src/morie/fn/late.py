# morie.fn -- function file (rootcoder007/morie)
"""Local average treatment effect by two-stage least squares."""

from __future__ import annotations

import math
from typing import Any

from ._qpcore import inverse, solve

__all__ = ["estimate_late"]


def _col(frame, name):
    v = frame[name]
    v = v.tolist() if hasattr(v, "tolist") else list(v)
    return [float(t) for t in v]


def _rss(Z, t):
    """Residual sum of squares of the least-squares fit of t on the columns of Z."""
    k = len(Z[0])
    A = [[math.fsum(r[a] * r[b] for r in Z) for b in range(k)] for a in range(k)]
    g = solve(A, [math.fsum(r[a] * v for r, v in zip(Z, t)) for a in range(k)])
    return math.fsum((v - math.fsum(c * x for c, x in zip(g, r))) ** 2 for r, v in zip(Z, t))


def estimate_late(
    data,
    *,
    treatment: str,
    outcome: str,
    instrument: str,
    covariates: list[str] | None = None,
    se_type: str = "homoskedastic",
) -> dict[str, Any]:
    r"""Local average treatment effect (LATE) by two-stage least squares.

    With a binary instrument ``Z`` and monotonicity the IV estimand is the
    effect among compliers (Imbens and Angrist 1994); without covariates
    2SLS is the Wald ratio ``Cov(Y, Z) / Cov(T, Z)``. With exogenous
    covariates ``W`` the second stage regresses ``Y`` on ``(1, W, T)``
    using instruments ``(1, W, Z)``: ``beta = (Xh'Xh)^{-1} Xh'y`` with
    ``Xh = Z (Z'Z)^{-1} Z'X``, residuals ``e = y - X beta`` (Wooldridge
    2010, sec. 5.2). Standard errors: ``"homoskedastic"`` uses ``s^2
    (Xh'Xh)^{-1}``, ``s^2 = e'e / (n - k)`` (the ``AER::ivreg`` default);
    ``"robust"`` is the HC0 sandwich ``(Xh'Xh)^{-1} Xh' diag(e^2) Xh
    (Xh'Xh)^{-1}``. ``f_stat`` is the first-stage F statistic for the
    excluded instrument, ``((RSS_r - RSS_u) / 1) / (RSS_u / (n - k_z))``.

    Parameters
    ----------
    data : DataFrame
        Input frame; rows with a missing value in a used column are dropped.
    treatment, outcome, instrument : str
        Endogenous treatment, outcome and instrument columns.
    covariates : list of str, optional
        Exogenous covariates.
    se_type : {"homoskedastic", "robust"}
        Covariance of the 2SLS estimator.

    Returns
    -------
    dict
        ``late``, ``se``, ``ci`` (normal 95%), ``f_stat``, ``n``,
        ``se_type``, ``method``.

    References
    ----------
    Imbens, G. W. and Angrist, J. D. (1994). Identification and estimation of local average
    treatment effects. *Econometrica*, 62(2), 467-475.

    Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed.
    MIT Press, ch. 5.

    Examples
    --------
    >>> d = {"z": [0, 0, 0, 0, 1, 1, 1, 1], "t": [0, 0, 1, 0, 1, 1, 0, 1],
    ...      "y": [1.0, 1.4, 3.1, 0.8, 3.3, 2.9, 1.2, 3.6]}
    >>> r = estimate_late(d, treatment="t", outcome="y", instrument="z")
    >>> round(r["late"], 10), round(r["se"], 10), round(r["f_stat"], 10)
    (2.35, 0.4354116826, 2.0)
    """
    if se_type not in ("homoskedastic", "robust"):
        raise ValueError("se_type must be 'homoskedastic' or 'robust'")
    covariates = list(covariates or [])
    cols = [treatment, outcome, instrument] + covariates
    raw = [_col(data, c) for c in cols]
    keep = [i for i in range(len(raw[0])) if not any(math.isnan(c[i]) for c in raw)]
    t, y, z = ([c[i] for i in keep] for c in raw[:3])
    W = [[c[i] for c in raw[3:]] for i in keep]
    n = len(keep)
    X = [[1.0] + w + [ti] for w, ti in zip(W, t)]
    Z = [[1.0] + w + [zi] for w, zi in zip(W, z)]
    k = len(X[0])
    if n <= k:
        raise ValueError(f"{n} complete rows cannot support {k} parameters")
    rss_u = _rss(Z, t)
    rss_r = _rss([r[:-1] for r in Z], t)
    if rss_r - rss_u <= 1e-12 * rss_r:
        raise ValueError(
            "Instrument has no partial correlation with treatment; LATE is not identified (weak instrument)."
        )
    f_stat = (rss_r - rss_u) / (rss_u / (n - k))
    ZtZi = inverse([[math.fsum(r[a] * r[b] for r in Z) for b in range(k)] for a in range(k)])
    ZtX = [[math.fsum(zr[a] * xr[b] for zr, xr in zip(Z, X)) for b in range(k)] for a in range(k)]
    Pi = [[math.fsum(ZtZi[a][c] * ZtX[c][b] for c in range(k)) for b in range(k)] for a in range(k)]
    Xh = [[math.fsum(zr[a] * Pi[a][b] for a in range(k)) for b in range(k)] for zr in Z]
    B = inverse([[math.fsum(r[a] * r[b] for r in Xh) for b in range(k)] for a in range(k)])
    Xhy = [math.fsum(r[a] * v for r, v in zip(Xh, y)) for a in range(k)]
    beta = [math.fsum(B[a][b] * Xhy[b] for b in range(k)) for a in range(k)]
    e = [v - math.fsum(c * x for c, x in zip(beta, r)) for r, v in zip(X, y)]
    j = k - 1
    if se_type == "homoskedastic":
        var = math.fsum(v * v for v in e) / (n - k) * B[j][j]
    else:
        g = [math.fsum(B[j][a] * r[a] for a in range(k)) for r in Xh]
        var = math.fsum((gi * ei) ** 2 for gi, ei in zip(g, e))
    late, se = beta[j], math.sqrt(var)
    zc = 1.959963984540054
    return {
        "late": late,
        "se": se,
        "ci": (late - zc * se, late + zc * se),
        "f_stat": f_stat,
        "n": n,
        "se_type": se_type,
        "method": "2SLS" + (" with covariates" if covariates else " (Wald ratio)"),
    }


late = estimate_late


def cheatsheet() -> str:
    return "estimate_late: 2SLS LATE, Y on (1, W, T) instrumented by (1, W, Z); homoskedastic or HC0 SE, first-stage F"


# compact alias per ledger/NAMING.md
estimatelate = estimate_late
