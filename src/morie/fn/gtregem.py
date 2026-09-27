"""Logistic regression from initial group-test responses by Xie's EM algorithm.

Bilder & Loughin (2025), Analysis of Categorical Data with R, eqs (6.32)-(6.33).
"""

import math

from ._richresult import RichResult
from .glmprofci import _solve

__all__ = ["gtregem"]


def gtregem(z, group, X, se=1.0, sp=1.0, max_iter=1000, tol=1e-10):
    r"""MLE of logit(pi_i) = x_i' beta (6.32) when only group responses Z_k are observed.

    E-step (Xie 2001): with P(Z_k = 1) = Se - (Se + Sp - 1) prod_{i in k}(1 - pi_i),
    E[Y_i | Z_k = 1] = Se pi_i / P(Z_k = 1) and
    E[Y_i | Z_k = 0] = (1 - Se) pi_i / (1 - P(Z_k = 1)); M-step: maximise the
    complete-data log-likelihood (6.33) with these fractional responses (a
    logistic regression by Newton steps). Iterate until the coefficients change
    by less than ``tol``. The observed-data log-likelihood never decreases.

    Parameters
    ----------
    z : sequence of 0/1
        Response of each group, indexed by group label.
    group : sequence of ints
        Group label (0..K-1) of each individual.
    X : n x p nested sequence
        Individual covariates including the intercept column.
    se, sp : float
        Sensitivity and specificity of the group test.

    Returns
    -------
    RichResult
        Keys: beta, loglik, iterations, converged, se_beta (from the observed information).

    References
    ----------
    Xie, M. (2001). Statistics in Medicine 20, 1957-1969.
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Eqs (6.32)-(6.33).

    Examples
    --------
    >>> xs = [((7 * i) % 13) / 4 - 1.5 for i in range(40)]
    >>> z = [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    >>> r = gtregem(z, [i // 2 for i in range(40)], [[1, x] for x in xs], se=0.95, sp=0.95)
    >>> r["converged"]
    True
    """
    X = [[float(v) for v in row] for row in X]
    group = [int(g) for g in group]
    z = [int(v) for v in z]
    n, p = len(X), len(X[0])
    K = len(z)
    if len(group) != n or min(group) < 0 or max(group) >= K or se + sp <= 1:
        raise ValueError("group labels must index z and Se + Sp > 1")
    members = [[] for _ in range(K)]
    for i, g in enumerate(group):
        members[g].append(i)

    def probs(b):
        return [1 / (1 + math.exp(-sum(a * c for a, c in zip(x, b)))) for x in X]

    def loglik(pi):
        ll = 0.0
        for k in range(K):
            q = 1.0
            for i in members[k]:
                q *= 1 - pi[i]
            pz = se - (se + sp - 1) * q
            ll += math.log(pz) if z[k] else math.log(1 - pz)
        return ll

    beta = [0.0] * p
    conv = False
    it = 0
    for _ in range(max_iter):
        it += 1
        pi = probs(beta)
        w = [0.0] * n
        for k in range(K):
            q = 1.0
            for i in members[k]:
                q *= 1 - pi[i]
            pz = se - (se + sp - 1) * q
            for i in members[k]:
                w[i] = se * pi[i] / pz if z[k] else (1 - se) * pi[i] / (1 - pz)
        b = beta[:]
        for _ in range(100):  # M-step: logistic regression with fractional responses
            pr = probs(b)
            g = [sum((w[i] - pr[i]) * X[i][a] for i in range(n)) for a in range(p)]
            H = [[sum(pr[i] * (1 - pr[i]) * X[i][a] * X[i][c] for i in range(n)) for c in range(p)] for a in range(p)]
            step = _solve(H, g)
            b = [u + s for u, s in zip(b, step)]
            if max(abs(s) for s in step) < 1e-13:
                break
        change = max(abs(u - v) for u, v in zip(b, beta))
        beta = b
        if change < tol:
            conv = True
            break
    pi = probs(beta)
    # observed information by central differences of the observed score
    h = 1e-5

    def score(b):
        return [
            (
                loglik(probs([v + (h if c == a else 0) for c, v in enumerate(b)]))
                - loglik(probs([v - (h if c == a else 0) for c, v in enumerate(b)]))
            )
            / (2 * h)
            for a in range(p)
        ]

    info = []
    for a in range(p):
        up = score([v + (h if c == a else 0) for c, v in enumerate(beta)])
        dn = score([v - (h if c == a else 0) for c, v in enumerate(beta)])
        info.append([-(u - d) / (2 * h) for u, d in zip(up, dn)])
    inv = [_solve(info, [float(a == c) for a in range(p)]) for c in range(p)]
    return RichResult(
        title="Group testing regression (Xie EM)",
        summary_lines=[("beta", beta), ("iterations", it)],
        payload={
            "beta": beta,
            "loglik": loglik(pi),
            "iterations": it,
            "converged": conv,
            "se_beta": [math.sqrt(max(0.0, inv[a][a])) for a in range(p)],
        },
    )


def cheatsheet():
    return "gtregem: logistic regression from group responses by Xie's EM. Bilder & Loughin eqs (6.32)-(6.33)."
