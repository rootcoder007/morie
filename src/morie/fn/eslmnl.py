"""Multinomial logistic regression by Newton-Raphson (ESL sec 4.4)."""

import math

from ._richresult import RichResult
from .nlsgn import _inverse

__all__ = ["esl_multinomial_logit"]


def _probs(Z, beta, K, q):
    out = []
    for z in Z:
        eta = [sum(z[j] * beta[k * q + j] for j in range(q)) for k in range(K - 1)]
        m = max(0.0, max(eta))
        e = [math.exp(v - m) for v in eta]
        den = math.exp(-m) + sum(e)
        out.append([v / den for v in e] + [math.exp(-m) / den])
    return out


def esl_multinomial_logit(X, g, query=None, max_iter=100, tol=1e-10):
    r"""Fit :math:`\log[\Pr(G=k\mid x)/\Pr(G=K\mid x)] = \beta_{k0} + \beta_k^Tx`, k = 1..K-1.

    The last class (in sorted order) is the baseline (ESL eqs 4.17-4.18).
    The log-likelihood :math:`\sum_i \log p_{g_i}(x_i;\theta)` is maximised by
    Newton-Raphson with step halving; the Hessian block for classes k, l is
    :math:`-\sum_i z_iz_i^T p_{ik}(\delta_{kl} - p_{il})` and the standard errors
    come from its inverse (for K = 2 this is the IRLS fit of eq 4.26).

    Parameters
    ----------
    X : N x p nested sequence
    g : sequence of N labels
    query : M x p nested sequence, optional
    max_iter, tol
        Newton controls (tol on the largest step).

    Returns
    -------
    RichResult
        ``coefficients`` and ``se`` ((K - 1) x (p + 1), intercept first),
        ``loglik``, ``classes``, ``baseline``, ``prob`` (rows of query),
        ``iterations``, ``converged``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 4.4.
    """
    labels = list(g)
    classes = sorted(set(labels), key=repr)
    K = len(classes)
    if K < 2:
        raise ValueError("need at least two classes")
    Z = [[1.0] + [float(v) for v in r] for r in X]
    n, q = len(Z), len(Z[0])
    if len(labels) != n:
        raise ValueError("X and g differ in length")
    yi = [classes.index(lab) for lab in labels]
    npar = (K - 1) * q
    beta = [0.0] * npar

    def loglik(b):
        P = _probs(Z, b, K, q)
        return sum(math.log(P[i][yi[i]]) for i in range(n)), P

    ll, P = loglik(beta)
    converged, it, info = False, 0, None
    for _ in range(max_iter):
        it += 1
        grad = [sum(Z[i][j] * ((yi[i] == k) - P[i][k]) for i in range(n)) for k in range(K - 1) for j in range(q)]
        info = [
            [
                sum(Z[i][a % q] * Z[i][b % q] * P[i][a // q] * ((a // q == b // q) - P[i][b // q]) for i in range(n))
                for b in range(npar)
            ]
            for a in range(npar)
        ]
        inv = _inverse(info)
        step = [sum(inv[a][b] * grad[b] for b in range(npar)) for a in range(npar)]
        fac = 1.0
        while fac >= 1 / 1024:
            cand = [b + fac * s for b, s in zip(beta, step)]
            llc, Pc = loglik(cand)
            if llc >= ll - 1e-12:
                beta, ll, P = cand, llc, Pc
                break
            fac /= 2
        if max(abs(fac * s) for s in step) < tol:
            converged = True
            break
    info = [
        [
            sum(Z[i][a % q] * Z[i][b % q] * P[i][a // q] * ((a // q == b // q) - P[i][b // q]) for i in range(n))
            for b in range(npar)
        ]
        for a in range(npar)
    ]
    inv = _inverse(info)
    coef = [[beta[k * q + j] for j in range(q)] for k in range(K - 1)]
    se = [[math.sqrt(inv[k * q + j][k * q + j]) for j in range(q)] for k in range(K - 1)]
    Qz = Z if query is None else [[1.0] + [float(v) for v in r] for r in query]
    return RichResult(
        title="Multinomial logistic regression",
        summary_lines=[("loglik", ll), ("classes", classes)],
        payload={
            "coefficients": coef,
            "se": se,
            "loglik": ll,
            "classes": classes,
            "baseline": classes[-1],
            "prob": _probs(Qz, beta, K, q),
            "iterations": it,
            "converged": converged,
        },
    )


def cheatsheet():
    return "eslmnl: Newton on the multinomial log-likelihood, last class as baseline; se from the inverse information"
