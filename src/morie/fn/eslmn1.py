"""L1-penalised multinomial logistic regression (ESL sec 18.4)."""

import math

from ._richresult import RichResult

__all__ = ["esl_multinomial_l1"]


def esl_multinomial_l1(X, g, lambda_, max_outer=500, max_inner=10000, tol=1e-12):
    r"""Maximise :math:`\sum_i\log\Pr(g_i|x_i) - \lambda\sum_k\sum_j|\beta_{kj}|` in the symmetric parametrisation.

    ESL eqs 18.10 and 18.19: :math:`\Pr(G=k|x) = e^{\beta_{k0}+x^T\beta_k}/\sum_\ell
    e^{\beta_{\ell0}+x^T\beta_\ell}` with all K coefficient vectors penalised
    (the lasso makes the symmetric form identifiable) and the intercepts
    free. Each outer pass cycles over the classes, forming the IRLS
    quadratic approximation for class k with the others fixed (weights
    :math:`p_{ik}(1-p_{ik})`) and minimising it by coordinate descent with soft
    thresholding, as ``glmnet``'s ungrouped multinomial. Intercepts are
    reported centred to sum to zero; this lambda is N times glmnet's.

    Parameters
    ----------
    X : n x p nested sequence
    g : sequence of n labels
    lambda_ : float
        Penalty, > 0.
    max_outer, max_inner, tol
        Iteration controls.

    Returns
    -------
    RichResult
        ``intercepts`` (K), ``coefficients`` (K x p), ``classes``,
        ``loglik``, ``objective``, ``prob``, ``converged``.

    References
    ----------
    Friedman, J., Hastie, T. & Tibshirani, R. (2010). JSS 33(1), sec. 4.
    """
    rows = [[float(v) for v in r] for r in X]
    labels = list(g)
    classes = sorted(set(labels), key=repr)
    n, p, K = len(rows), len(rows[0]), len(classes)
    lam = float(lambda_)
    if len(labels) != n or K < 2 or lam <= 0:
        raise ValueError("need matching X and g, at least two classes and lambda > 0")
    Y = [[1.0 if lab == c else 0.0 for c in classes] for lab in labels]
    b0 = [0.0] * K
    B = [[0.0] * p for _ in range(K)]

    def probs():
        out = []
        for r in rows:
            eta = [b0[k] + sum(B[k][j] * r[j] for j in range(p)) for k in range(K)]
            m = max(eta)
            e = [math.exp(v - m) for v in eta]
            s = sum(e)
            out.append([v / s for v in e])
        return out

    converged = False
    for _ in range(max_outer):
        old = [v for k in range(K) for v in [b0[k]] + B[k]]
        for k in range(K):
            P = probs()
            w = [max(P[i][k] * (1 - P[i][k]), 1e-10) for i in range(n)]
            r = [(Y[i][k] - P[i][k]) / w[i] for i in range(n)]  # working residual z - eta
            xw2 = [sum(w[i] * rows[i][j] ** 2 for i in range(n)) for j in range(p)]
            sw = sum(w)
            for _ in range(max_inner):
                d0 = sum(w[i] * r[i] for i in range(n)) / sw
                b0[k] += d0
                r = [ri - d0 for ri in r]
                delta = abs(d0)
                for j in range(p):
                    rho = sum(w[i] * rows[i][j] * r[i] for i in range(n)) + xw2[j] * B[k][j]
                    new = math.copysign(max(abs(rho) - lam, 0.0), rho) / xw2[j] if xw2[j] > 0 else 0.0
                    if new != B[k][j]:
                        dj = new - B[k][j]
                        r = [ri - rows[i][j] * dj for i, ri in enumerate(r)]
                        B[k][j] = new
                        delta = max(delta, abs(dj))
                if delta < tol:
                    break
        new = [v for k in range(K) for v in [b0[k]] + B[k]]
        if max(abs(a - b) for a, b in zip(new, old)) < tol:
            converged = True
            break
    c = sum(b0) / K
    b0 = [v - c for v in b0]
    P = probs()
    ll = sum(math.log(P[i][classes.index(labels[i])]) for i in range(n))
    return RichResult(
        title="L1-penalised multinomial logistic regression",
        summary_lines=[("loglik", ll), ("classes", classes)],
        payload={
            "intercepts": b0,
            "coefficients": B,
            "classes": classes,
            "loglik": ll,
            "objective": -ll + lam * sum(abs(v) for row in B for v in row),
            "prob": P,
            "converged": converged,
        },
    )


def cheatsheet():
    return "eslmn1: per-class IRLS + lasso coordinate descent (glmnet ungrouped multinomial); lambda = N x glmnet's"
