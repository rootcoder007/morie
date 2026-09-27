"""L1-penalised logistic regression by proximal Newton and coordinate descent (ESL sec 4.4.4)."""

import math

from ._richresult import RichResult

__all__ = ["esl_l1_logistic"]


def esl_l1_logistic(X, y, lambda_, max_outer=200, max_inner=10000, tol=1e-12):
    r"""Maximise :math:`\sum_i [y_i(\beta_0+\beta^Tx_i) - \log(1+e^{\beta_0+\beta^Tx_i})] - \lambda\sum_j|\beta_j|`.

    ESL eq 4.31: the intercept is not penalised. Each outer step forms the
    IRLS quadratic approximation (weights :math:`w_i = p_i(1-p_i)`, working
    response :math:`z_i = \eta_i + (y_i-p_i)/w_i`) and minimises
    :math:`\tfrac12\sum_i w_i(z_i - \beta_0 - \beta^Tx_i)^2 + \lambda\|\beta\|_1` by
    cyclic coordinate descent with soft thresholding. At the solution the
    active coefficients satisfy :math:`x_j^T(y - p) = \lambda\,\mathrm{sign}(\beta_j)`
    (eq 4.32). This lambda is N times glmnet's.

    Parameters
    ----------
    X : N x p nested sequence
    y : sequence of 0/1
    lambda_ : float
        Penalty, >= 0.
    max_outer, max_inner, tol
        Iteration controls.

    Returns
    -------
    RichResult
        ``intercept``, ``beta``, ``active_set`` (0-based), ``loglik``,
        ``objective``, ``score`` (:math:`x_j^T(y - p)`), ``converged``.

    References
    ----------
    Friedman, J., Hastie, T. & Tibshirani, R. (2010). Regularization paths for
    generalized linear models via coordinate descent. JSS 33(1).
    """
    Xm = [[float(v) for v in r] for r in X]
    yy = [float(v) for v in y]
    n, p = len(Xm), len(Xm[0])
    lam = float(lambda_)
    if len(yy) != n or any(v not in (0.0, 1.0) for v in yy) or lam < 0:
        raise ValueError("need 0/1 responses matching X and lambda >= 0")
    b0, b = 0.0, [0.0] * p

    def eta_p():
        eta = [b0 + sum(Xm[i][j] * b[j] for j in range(p)) for i in range(n)]
        return eta, [1 / (1 + math.exp(-e)) for e in eta]

    converged = False
    for _ in range(max_outer):
        eta, pr = eta_p()
        w = [max(v * (1 - v), 1e-10) for v in pr]
        z = [e + (yv - v) / wv for e, yv, v, wv in zip(eta, yy, pr, w)]
        old = [b0] + b[:]
        r = [z[i] - eta[i] for i in range(n)]
        xw2 = [sum(w[i] * Xm[i][j] ** 2 for i in range(n)) for j in range(p)]
        sw = sum(w)
        for _ in range(max_inner):
            delta = 0.0
            d0 = sum(w[i] * r[i] for i in range(n)) / sw
            b0 += d0
            r = [ri - d0 for ri in r]
            delta = abs(d0)
            for j in range(p):
                rho = sum(w[i] * Xm[i][j] * r[i] for i in range(n)) + xw2[j] * b[j]
                new = math.copysign(max(abs(rho) - lam, 0.0), rho) / xw2[j] if xw2[j] > 0 else 0.0
                if new != b[j]:
                    dj = new - b[j]
                    r = [ri - Xm[i][j] * dj for i, ri in enumerate(r)]
                    b[j] = new
                    delta = max(delta, abs(dj))
            if delta < tol:
                break
        if max(abs(u - v) for u, v in zip([b0] + b, old)) < tol:
            converged = True
            break
    eta, pr = eta_p()
    ll = sum(
        yv * e - math.log1p(math.exp(e)) if e < 30 else yv * e - e - math.log1p(math.exp(-e)) for yv, e in zip(yy, eta)
    )
    score = [sum(Xm[i][j] * (yy[i] - pr[i]) for i in range(n)) for j in range(p)]
    return RichResult(
        title="L1-penalised logistic regression",
        summary_lines=[("intercept", b0), ("beta", b)],
        payload={
            "intercept": b0,
            "beta": b,
            "active_set": [j for j in range(p) if b[j] != 0],
            "loglik": ll,
            "objective": -ll + lam * sum(abs(v) for v in b),
            "score": score,
            "converged": converged,
        },
    )


def cheatsheet():
    return "esll1l: proximal Newton (IRLS quadratic + lasso coordinate descent); lambda = N x glmnet's"
