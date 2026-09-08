# morie.fn -- function file (rootcoder007/morie)
"""Cox proportional-hazards partial likelihood, Breslow handling of ties.

Sources CONSULTED: Cox, D. R. (1972), "Regression models and life
tables (with discussion)", *JRSS B* 34(2):187-220, and Cox, D. R.
(1975), "Partial likelihood", *Biometrika* 62(2):269-276.  Both are
paywalled and could not be retrieved; the expressions implemented are
the standard published statement of the partial likelihood with the
Breslow approximation for tied event times.  Writing eta_j = x_j'beta,
w_j = exp(eta_j), and for each distinct event time t

    R(t) = { j : time_j >= t }        the risk set
    D(t) = { i : time_i = t, event }  the tied deaths, d = |D(t)|
    S0 = sum_{j in R} w_j
    S1 = sum_{j in R} w_j x_j          (a vector)
    S2 = sum_{j in R} w_j x_j x_j'     (a matrix)

the log partial likelihood, score and observed information are

    l(beta)  = sum_t [ sum_{i in D(t)} eta_i - d log S0 ]
    U(beta)  = sum_t [ sum_{i in D(t)} x_i - d S1/S0 ]
    I(beta)  = sum_t   d [ S2/S0 - (S1/S0)(S1/S0)' ].

VERIFIED numerically against ``survival::coxph(..., ties = "breslow")``
-- Therneau's implementation, the reference for this model in R.  The
parity harness checks the log partial likelihood at a fixed beta against
``fit$loglik[1]`` with ``iter.max = 0``, and the fitted coefficients
from the Newton-Raphson branch below against ``coef(fit)``.

Efron's approximation for ties is NOT implemented; the Breslow form is
the one Cox states, and mixing the two silently under one name would be
worse than not offering the second.
"""

import math

from ._richresult import RichResult

__all__ = ["cox_partial_likelihood"]


def _inv(A):
    """Inverse by Gauss-Jordan with partial pivoting."""
    k = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(k)]
         for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            raise ValueError("information matrix is singular")
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(k):
            if r == c:
                continue
            fac = M[r][c] / pv
            if fac == 0.0:
                continue
            for t in range(c, 2 * k):
                M[r][t] -= fac * M[c][t]
    return [[M[i][k + j] / M[i][i] for j in range(k)] for i in range(k)]


def _terms(time, event, X, beta):
    """Log partial likelihood, score and observed information at beta."""
    n = len(time)
    p = len(beta)
    eta = [sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    w = [math.exp(e) for e in eta]
    order = sorted(range(n), key=lambda i: (-time[i], i))
    loglik = 0.0
    score = [0.0] * p
    info = [[0.0] * p for _ in range(p)]
    s0 = 0.0
    s1 = [0.0] * p
    s2 = [[0.0] * p for _ in range(p)]
    k = 0
    nevent = 0
    while k < n:
        t = time[order[k]]
        j = k
        while j < n and time[order[j]] == t:
            i = order[j]
            s0 += w[i]
            for a in range(p):
                s1[a] += w[i] * X[i][a]
                for b in range(p):
                    s2[a][b] += w[i] * X[i][a] * X[i][b]
            j += 1
        deaths = [order[m] for m in range(k, j) if event[order[m]]]
        d = len(deaths)
        if d:
            nevent += d
            for i in deaths:
                loglik += eta[i]
                for a in range(p):
                    score[a] += X[i][a]
            loglik -= d * math.log(s0)
            for a in range(p):
                score[a] -= d * s1[a] / s0
                for b in range(p):
                    info[a][b] += d * (s2[a][b] / s0
                                       - (s1[a] / s0) * (s1[b] / s0))
        k = j
    return loglik, score, info, nevent


def cox_partial_likelihood(time, event, X, beta=None, max_iter=50,
                           tol=1e-10):
    """Cox partial likelihood, its score and information, optionally fitted.

    Parameters
    ----------
    time : sequence, length n
        Follow-up times.
    event : sequence, length n
        1 (or True) for an observed event, 0 for right censoring.
    X : sequence of sequences, shape (n, p)
        Covariates.  No intercept: the baseline hazard absorbs it, and a
        constant column would make the information matrix singular.
    beta : sequence, length p, optional
        Coefficients at which to evaluate.  When omitted the partial
        likelihood is maximised by Newton-Raphson from zero and the
        returned quantities are those at the maximum.
    max_iter, tol : int, float
        Newton-Raphson controls, used only when ``beta`` is omitted.

    Returns
    -------
    RichResult
        Keys ``loglik``, ``score``, ``information``, ``coefficients``,
        ``se``, ``vcov``, ``n``, ``n_event``, ``iterations``,
        ``converged``, ``method``.
    """
    time = [float(t) for t in time]
    event = [1 if bool(e) else 0 for e in event]
    X = [[float(v) for v in row] for row in X]
    n = len(time)
    if len(event) != n or len(X) != n:
        raise ValueError("time, event and X must have the same length")
    if n == 0:
        raise ValueError("no observations")
    p = len(X[0])
    for row in X:
        if len(row) != p:
            raise ValueError("every row of X needs the same number of columns")
    if not any(event):
        raise ValueError("no events; the partial likelihood is empty")

    fitted = beta is None
    iterations = 0
    converged = True
    if fitted:
        b = [0.0] * p
        converged = False
        for iterations in range(1, max_iter + 1):
            _ll, u, info = _terms(time, event, X, b)[:3]
            step = _inv(info)
            delta = [sum(step[a][c] * u[c] for c in range(p)) for a in range(p)]
            b = [b[a] + delta[a] for a in range(p)]
            if max(abs(v) for v in delta) < tol:
                converged = True
                break
        beta = b
    else:
        beta = [float(v) for v in beta]
        if len(beta) != p:
            raise ValueError("beta must have one entry per column of X")

    loglik, score, info, nevent = _terms(time, event, X, beta)
    vcov = _inv(info)
    return RichResult(
        payload={
            "loglik": loglik,
            "score": score,
            "information": info,
            "coefficients": list(beta),
            "vcov": vcov,
            "se": [math.sqrt(vcov[a][a]) for a in range(p)],
            "n": n,
            "n_event": nevent,
            "iterations": iterations,
            "converged": converged,
            "method": "Cox partial likelihood, Breslow ties"
            + (" (Newton-Raphson fit)" if fitted else " (evaluated at beta)"),
        }
    )


def cheatsheet():
    return "survcox: Cox partial likelihood evaluation"
