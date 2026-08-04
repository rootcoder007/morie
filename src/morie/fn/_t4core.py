# morie.fn -- shared helpers for the tail4 batch
"""Shared numeric helpers for the tail4 batch.

Internal only.  The R mirror is ``R/aaa_tail4_core.R``; every routine
here exists in both arms so the two can be compared value-for-value.
Nothing in this file is exported.
"""

from __future__ import annotations

import math

__all__ = []


def vec(x):
    """Flatten ``x`` to a plain list of floats."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    out = []
    if not isinstance(x, (list, tuple)):
        return [float(x)]
    for item in x:
        if isinstance(item, (list, tuple)):
            out.extend(vec(item))
        elif hasattr(item, "tolist"):
            out.extend(vec(item.tolist()))
        else:
            out.append(float(item))
    return out


def mat(X):
    """Coerce ``X`` to a list of equal-length lists of floats."""
    if hasattr(X, "tolist"):
        X = X.tolist()
    if not isinstance(X, (list, tuple)):
        return [[float(X)]]
    rows = []
    for r in X:
        if isinstance(r, (list, tuple)) or hasattr(r, "tolist"):
            rows.append(vec(r))
        else:
            rows.append([float(r)])
    return rows


def ranks(x):
    """Midranks (average method), matching R ``rank()``."""
    n = len(x)
    order = sorted(range(n), key=lambda i: (x[i], i))
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def tiecounts(x):
    """Multiplicities of the distinct values of ``x`` (R ``table()``)."""
    seen = {}
    for v in x:
        seen[v] = seen.get(v, 0) + 1
    return [seen[k] for k in sorted(seen)]


def acfbiased(x, lag):
    """Sample autocorrelations r_1..r_lag, R ``acf()`` normalisation.

    Both numerator and denominator divide by ``n`` (the biased, positive
    semi-definite estimator), which is what ``stats::acf`` and therefore
    ``stats::Box.test`` use.
    """
    n = len(x)
    xbar = sum(x) / n
    d = [xi - xbar for xi in x]
    c0 = sum(di * di for di in d)
    out = []
    for k in range(1, lag + 1):
        ck = sum(d[t] * d[t - k] for t in range(k, n))
        out.append(ck / c0 if c0 > 0 else float("nan"))
    return out


def lrvnw(u, lag):
    """Newey-West long-run variance of ``u`` with Bartlett weights.

    ``s^2 + (2/n) sum_{i=1}^{l} (1 - i/(l+1)) sum_{t>i} u_t u_{t-i}``.
    This is ``tseries``'s ``pp_sum`` C routine, which is what
    ``tseries::pp.test`` calls to build lambda^2.
    """
    n = len(u)
    s = sum(ui * ui for ui in u) / n
    tot = 0.0
    for i in range(1, lag + 1):
        acc = 0.0
        for t in range(i, n):
            acc += u[t] * u[t - i]
        tot += acc * (1.0 - i / (lag + 1.0))
    return s + 2.0 * tot / n


def olsfit(X, y):
    """Least squares by Gaussian elimination on the normal equations.

    Returns ``(beta, fitted, resid, xtxinv)``.  Deterministic and
    dependency-free; ``X`` is a list of rows already carrying whatever
    intercept column the caller wants.
    """
    n = len(X)
    p = len(X[0])
    xtx = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    aug = [xtx[a][:] + [1.0 if b == a else 0.0 for b in range(p)] + [xty[a]] for a in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(aug[r][c]))
        if abs(aug[piv][c]) < 1e-300:
            raise ValueError("singular design matrix")
        aug[c], aug[piv] = aug[piv], aug[c]
        d = aug[c][c]
        aug[c] = [v / d for v in aug[c]]
        for r in range(p):
            if r == c:
                continue
            f = aug[r][c]
            if f != 0.0:
                aug[r] = [aug[r][k] - f * aug[c][k] for k in range(len(aug[r]))]
    xtxinv = [row[p:2 * p] for row in aug]
    beta = [row[2 * p] for row in aug]
    fitted = [sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    resid = [y[i] - fitted[i] for i in range(n)]
    return beta, fitted, resid, xtxinv


def kendallS(x, y):
    """Concordant minus discordant pairs, with tie counts.

    Returns ``(S, n, tx, ty)`` where ``tx``/``ty`` are the multiplicity
    lists of the tied groups in ``x`` and ``y``.
    """
    n = len(x)
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            a = x[j] - x[i]
            b = y[j] - y[i]
            sgn = (1 if a > 0 else (-1 if a < 0 else 0)) * (1 if b > 0 else (-1 if b < 0 else 0))
            S += sgn
    return S, n, tiecounts(x), tiecounts(y)


def kendalltaub(x, y):
    """Kendall's tau-b and the two-sided normal-approximation p-value.

    Matches ``stats::cor.test(method = "kendall", exact = FALSE)``: the
    statistic is ``S / sqrt(v)`` with the tie-corrected variance of
    Kendall (1970, sec. 3.5), and no continuity correction.
    """
    S, n, tx, ty = kendallS(x, y)
    n0 = n * (n - 1) / 2.0
    n1 = sum(t * (t - 1) for t in tx) / 2.0
    n2 = sum(t * (t - 1) for t in ty) / 2.0
    den = math.sqrt((n0 - n1) * (n0 - n2))
    tau = S / den if den > 0 else float("nan")
    v0 = n * (n - 1) * (2.0 * n + 5)
    vt = sum(t * (t - 1) * (2.0 * t + 5) for t in tx)
    vu = sum(t * (t - 1) * (2.0 * t + 5) for t in ty)
    v1 = sum(t * (t - 1) for t in tx) * sum(t * (t - 1) for t in ty)
    v2 = sum(t * (t - 1) * (t - 2) for t in tx) * sum(t * (t - 1) * (t - 2) for t in ty)
    v = ((v0 - vt - vu) / 18.0
         + v1 / (2.0 * n * (n - 1))
         + v2 / (9.0 * n * (n - 1) * (n - 2)))
    z = S / math.sqrt(v) if v > 0 else float("nan")
    return tau, z


# pnorm / pchisq deliberately absent: ``_stats_core.norm`` and
# ``_stats_core.chi2`` already agree with R's own pnorm/pchisq to 1e-16
# relative, so this batch reuses them rather than carrying a second
# implementation.


def result(**kw):
    """Build the batch's RichResult, keeping the import in one place."""
    from ._richresult import RichResult

    return RichResult(payload=kw)
