"""Proportional odds (cumulative logit) model fitted by maximum likelihood.

Bilder & Loughin (2025), Analysis of Categorical Data with R, eqs (3.11), (3.15).
"""

import math

from ._richresult import RichResult
from .glmprofci import _solve

__all__ = ["polrfit"]


def _F(c):
    if c == math.inf:
        return 1.0
    if c == -math.inf:
        return 0.0
    return 1 / (1 + math.exp(-c)) if c >= 0 else math.exp(c) / (1 + math.exp(c))


def _f(c):
    if math.isinf(c):
        return 0.0
    p = _F(c)
    return p * (1 - p)


def polrfit(y, X, max_iter=200, tol=1e-12):
    r"""logit P(Y <= j) = beta_j0 + x' beta, j = 1, ..., J - 1 (3.11, 3.15), by Newton-Raphson.

    Each observation contributes log(F(c_j) - F(c_{j-1})) with
    c_j = beta_j0 + x' beta, F logistic; the analytic gradient and Hessian use
    f = F(1 - F) and f' = f(1 - 2F). Steps are halved until the
    log-likelihood increases. MASS::polr reports zeta_j = beta_j0 and
    coefficients -beta.

    Parameters
    ----------
    y : sequence of int
        Ordered categories coded 0, ..., J - 1 (all present).
    X : n x p nested sequence
        Explanatory variables without an intercept column.

    Returns
    -------
    RichResult
        Keys: intercepts (beta_j0), beta, se_intercepts, se_beta, loglik, iterations.

    References
    ----------
    McCullagh, P. (1980). JRSS B 42, 109-142.
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Eqs (3.11), (3.15).

    Examples
    --------
    >>> r = polrfit([0, 0, 1, 0, 1, 2, 1, 2, 2, 1], [[x] for x in range(10)])
    >>> r["beta"][0] < 0
    True
    """
    y = [int(v) for v in y]
    X = [[float(v) for v in row] for row in X]
    n = len(y)
    J = max(y) + 1
    p = len(X[0]) if X else 0
    if J < 2 or sorted(set(y)) != list(range(J)) or len(X) != n:
        raise ValueError("y must use every category 0..J-1 and match X")
    k = J - 1
    cum = [sum(1 for v in y if v <= j) / n for j in range(k)]
    par = [math.log(c / (1 - c)) for c in cum] + [0.0] * p

    def cuts(par, i):
        eta = sum(a * b for a, b in zip(X[i], par[k:]))
        j = y[i]
        hi = par[j] + eta if j < k else math.inf
        lo = par[j - 1] + eta if j > 0 else -math.inf
        return hi, lo

    def loglik(par):
        if any(par[j + 1] <= par[j] for j in range(k - 1)):
            return -math.inf
        s = 0.0
        for i in range(n):
            hi, lo = cuts(par, i)
            pr = _F(hi) - _F(lo)
            if pr <= 0:
                return -math.inf
            s += math.log(pr)
        return s

    def derivs(par):
        m = k + p
        g = [0.0] * m
        H = [[0.0] * m for _ in range(m)]
        for i in range(n):
            hi, lo = cuts(par, i)
            P = _F(hi) - _F(lo)
            fa, fb = _f(hi), _f(lo)
            da, db = fa / P, -fb / P
            faa = (fa * (1 - 2 * _F(hi)) / P - fa * fa / P / P) if not math.isinf(hi) else 0.0
            fbb = (-fb * (1 - 2 * _F(lo)) / P - fb * fb / P / P) if not math.isinf(lo) else 0.0
            fab = fa * fb / P / P
            j = y[i]
            va = [0.0] * m
            vb = [0.0] * m
            if j < k:
                va[j] = 1.0
            if j > 0:
                vb[j - 1] = 1.0
            for c in range(p):
                va[k + c] = X[i][c] if j < k else 0.0
                vb[k + c] = X[i][c] if j > 0 else 0.0
            for a in range(m):
                g[a] += da * va[a] + db * vb[a]
                for b in range(m):
                    H[a][b] += faa * va[a] * va[b] + fbb * vb[a] * vb[b] + fab * (va[a] * vb[b] + vb[a] * va[b])
        return g, H

    ll = loglik(par)
    it = 0
    for _ in range(max_iter):
        it += 1
        g, H = derivs(par)
        step = _solve([[-v for v in row] for row in H], g)
        t = 1.0
        while True:
            cand = [a + t * s for a, s in zip(par, step)]
            lc = loglik(cand)
            if lc >= ll or t < 1e-12:
                break
            t /= 2
        done = max(abs(t * s) for s in step) < tol * (1 + max(abs(v) for v in par))
        par, ll = cand, lc
        if done:
            break
    g, H = derivs(par)
    m = k + p
    inv = [_solve([[-v for v in row] for row in H], [float(a == b) for a in range(m)]) for b in range(m)]
    se = [math.sqrt(inv[a][a]) for a in range(m)]
    return RichResult(
        title="Proportional odds model",
        summary_lines=[("beta", par[k:]), ("loglik", ll)],
        payload={
            "intercepts": par[:k],
            "beta": par[k:],
            "se_intercepts": se[:k],
            "se_beta": se[k:],
            "loglik": ll,
            "iterations": it,
        },
    )


def cheatsheet():
    return (
        "polrfit: proportional odds (cumulative logit) ML fit by Newton-Raphson. Bilder & Loughin eqs (3.11), (3.15)."
    )
