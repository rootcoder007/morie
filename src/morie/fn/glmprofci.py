"""Profile likelihood-ratio interval for a GLM coefficient (binomial logit or Poisson log).

Bilder & Loughin (2025), Analysis of Categorical Data with R, eqs (2.12)-(2.13).
"""

import math

from ._richresult import RichResult
from ._rrng_core import qchisq

__all__ = ["glmprofci"]


def _solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[piv][k]) < 1e-300:
            raise ValueError("singular information matrix")
        M[k], M[piv] = M[piv], M[k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            if f:
                for j in range(k, n + 1):
                    M[i][j] -= f * M[k][j]
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        x[k] = (M[k][n] - sum(M[k][j] * x[j] for j in range(k + 1, n))) / M[k][k]
    return x


def _irls(y, X, family, trials, offset, max_iter=100, tol=1e-12):
    """MLE by Fisher scoring; returns beta, fitted means, log-likelihood (up to constants) and the information."""
    n, p = len(y), len(X[0]) if X else 0
    beta = [0.0] * p
    if family == "binomial":
        eta = [math.log((yi + 0.5) / (ni - yi + 0.5)) for yi, ni in zip(y, trials)]
    else:
        eta = [math.log(yi + 0.5) for yi in y]
    for _ in range(max_iter):
        if family == "binomial":
            pr = [1 / (1 + math.exp(-e)) for e in eta]
            mu = [ni * q for ni, q in zip(trials, pr)]
            w = [ni * q * (1 - q) for ni, q in zip(trials, pr)]
        else:
            mu = [math.exp(e) for e in eta]
            w = mu[:]
        z = [e - o + (yi - m) / wi for e, o, yi, m, wi in zip(eta, offset, y, mu, w)]
        if p:
            A = [[sum(w[i] * X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
            rhs = [sum(w[i] * X[i][a] * z[i] for i in range(n)) for a in range(p)]
            new = _solve(A, rhs)
        else:
            new = []
        eta = [o + sum(xi[a] * new[a] for a in range(p)) for xi, o in zip(X, offset)]
        done = p == 0 or max(abs(a - b) for a, b in zip(new, beta)) < tol * (1 + max(abs(v) for v in new))
        beta = new
        if done:
            break
    if family == "binomial":
        pr = [1 / (1 + math.exp(-e)) for e in eta]
        mu = [ni * q for ni, q in zip(trials, pr)]
        ll = sum(yi * math.log(q) + (ni - yi) * math.log(1 - q) for yi, ni, q in zip(y, trials, pr))
        w = [ni * q * (1 - q) for ni, q in zip(trials, pr)]
    else:
        mu = [math.exp(e) for e in eta]
        ll = sum(yi * e - m for yi, e, m in zip(y, eta, mu))
        w = mu[:]
    info = [[sum(w[i] * X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    return beta, mu, ll, info


def _prep(y, X, family, trials):
    y = [float(v) for v in y]
    X = [[float(v) for v in row] for row in X]
    if family not in ("binomial", "poisson"):
        raise ValueError('family must be "binomial" or "poisson"')
    trials = [1.0] * len(y) if trials is None else [float(v) for v in trials]
    if len(X) != len(y) or len(trials) != len(y):
        raise ValueError("y, X and trials must have matching rows")
    return y, X, trials


def glmprofci(y, X, j, family="binomial", trials=None, alpha=0.05):
    r"""Profile LR interval for coefficient j: {b : -2 log(L(beta_tilde(b), b) / L(beta_hat)) < chi2_{1,1-alpha}}.

    The other coefficients are re-maximised for each fixed b (a GLM with offset
    x_j b), and the two limits are the roots of (2.13) on either side of the
    MLE, found by bracketing from the Wald interval and bisection to 1e-10.

    Parameters
    ----------
    y : sequence
        Successes (binomial) or counts (Poisson).
    X : n x p nested sequence
        Design matrix including any intercept column.
    j : int
        Coefficient index.
    family : {"binomial", "poisson"}
    trials : sequence, optional
        Binomial trials per row (default 1).
    alpha : float

    Returns
    -------
    RichResult
        Keys: estimate, se, ci, wald_ci, beta.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Eqs (2.12)-(2.13).

    Examples
    --------
    >>> r = glmprofci([0, 0, 1, 0, 1, 1, 1, 0, 1, 1], [[1, x] for x in range(10)], 1)
    >>> r["ci"][0] < r["estimate"] < r["ci"][1]
    True
    """
    y, X, trials = _prep(y, X, family, trials)
    p = len(X[0])
    beta, _, ll, info = _irls(y, X, family, trials, [0.0] * len(y))
    col = [_solve(info, [float(a == b) for a in range(p)]) for b in range(p)]
    se = math.sqrt(col[j][j])
    crit = qchisq(1 - alpha, 1)
    others = [[v for a, v in enumerate(row) if a != j] for row in X]

    def lr(b):
        off = [row[j] * b for row in X]
        _, _, llb, _ = _irls(y, others, family, trials, off)
        return 2 * (ll - llb) - crit

    lims = []
    for sgn in (-1, 1):
        step = se * math.sqrt(crit)
        far = beta[j] + sgn * step
        k = 0
        while lr(far) < 0:
            step *= 2
            far = beta[j] + sgn * step
            k += 1
            if k > 60:
                raise ValueError("profile interval is unbounded on one side")
        lo, hi = (far, beta[j]) if sgn < 0 else (beta[j], far)
        for _ in range(200):
            mid = (lo + hi) / 2
            inside = lr(mid) < 0
            if sgn < 0:
                lo, hi = (lo, mid) if inside else (mid, hi)
            else:
                lo, hi = (mid, hi) if inside else (lo, mid)
            if hi - lo < 1e-10 * (1 + abs(mid)):
                break
        lims.append((lo + hi) / 2)
    z = math.sqrt(crit)
    return RichResult(
        title="Profile likelihood interval",
        summary_lines=[("estimate", beta[j]), ("ci", tuple(lims))],
        payload={
            "estimate": beta[j],
            "se": se,
            "ci": tuple(lims),
            "wald_ci": (beta[j] - z * se, beta[j] + z * se),
            "beta": beta,
        },
    )


def cheatsheet():
    return "glmprofci: profile LR interval for a logistic or Poisson coefficient. Bilder & Loughin eqs (2.12)-(2.13)."
