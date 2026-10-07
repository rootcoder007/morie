"""Panel binary choice: conditional (fixed-effects) logit and random-effects logit/probit.

Chamberlain, G. (1980). Analysis of covariance with qualitative data. Review of Economic
Studies 47, 225-238. Butler, J. S. and Moffitt, R. (1982). A computationally efficient
quadrature procedure for the one-factor multinomial probit model. Econometrica 50, 761-764.
"""

import math

from ._qncore import dot
from ._richresult import RichResult
from .bfgsmin import bfgs_minimize

__all__ = ["panel_binary_choice"]


def _gh(n):
    """Gauss-Hermite nodes and weights for the weight exp(-x^2) (Golub-Welsch via symmetric QR)."""
    from ._mlfa import eigh_desc

    J = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        J[i][i + 1] = J[i + 1][i] = math.sqrt((i + 1) / 2)
    vals, vecs = eigh_desc(J)
    nodes = vals
    weights = [math.sqrt(math.pi) * vecs[0][k] ** 2 for k in range(n)]
    return nodes, weights


def _phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def panel_binary_choice(y, X, group, model="fe_logit", n_quad=30):
    r"""Binary panel models for y_it in {0, 1} with covariates x_it and unit effects a_i.

    ``"fe_logit"``: Chamberlain's conditional logit, maximising
    prod_i exp(sum_t y_it x_it b) / sum_{d: sum d = sum y_i} exp(sum_t d_t x_it b), which removes
    a_i; the denominator is built by dynamic programming over t. Units with all zeros or all
    ones carry no information and drop out. ``"re_logit"``/``"re_probit"``: a_i ~ N(0, s^2) with
    an intercept, marginal likelihood prod_i int prod_t F((2 y_it - 1)(x_it b + s u)) phi(u) du by
    Gauss-Hermite quadrature with ``n_quad`` nodes (Butler and Moffitt 1982). Both are maximised
    by BFGS; standard errors come from the inverse numerical Hessian of the log-likelihood.

    Parameters
    ----------
    y : sequence of 0/1
    X : list of covariate vectors (no intercept column)
    group : sequence of unit labels
    model : {"fe_logit", "re_logit", "re_probit"}
    n_quad : int

    Returns
    -------
    RichResult
        Keys: coef (intercept first for the random-effects models), se, sigma (random-effects
        SD), loglik, n_units, converged.

    References
    ----------
    Chamberlain, G. (1980). Review of Economic Studies 47, 225-238.
    Butler, J. S. and Moffitt, R. (1982). Econometrica 50, 761-764.

    Examples
    --------
    >>> r = panel_binary_choice([0, 1, 1, 0, 0, 1], [[0.0], [1.0], [2.0], [0.5], [0.2], [1.5]], [1, 1, 1, 2, 2, 2])
    >>> r["n_units"]
    2
    """
    Y = [int(v) for v in y]
    Xm = [[float(t) for t in r] for r in X]
    k = len(Xm[0])
    units = {}
    for i, g in enumerate(group):
        units.setdefault(g, []).append(i)
    if model == "fe_logit":
        keep = [idx for idx in units.values() if 0 < sum(Y[i] for i in idx) < len(idx)]

        def nll(b):
            tot = 0.0
            for idx in keep:
                eta = [dot(Xm[i], b) for i in idx]
                s = sum(Y[i] for i in idx)
                m = max(eta)
                # DP over t: D[j] = sum over subsets of size j of exp(sum eta - j m)
                D = [1.0] + [0.0] * s
                for e in eta:
                    w = math.exp(e - m)
                    for j in range(min(s, len(D) - 1), 0, -1):
                        D[j] += D[j - 1] * w
                num = sum(eta[t] for t, i in enumerate(idx) if Y[i]) - s * m
                tot -= num - math.log(D[s])
            return tot

        start = [0.0] * k
        n_units = len(keep)
    elif model in ("re_logit", "re_probit"):
        nodes, wts = _gh(int(n_quad))

        def F(z):
            if model == "re_logit":
                return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            return _phi(z)

        def nll(th):
            b0, b, ls = th[0], th[1 : 1 + k], th[1 + k]
            s = math.exp(ls)
            tot = 0.0
            for idx in units.values():
                eta = [b0 + dot(Xm[i], b) for i in idx]
                acc = 0.0
                for z, w in zip(nodes, wts):
                    u = math.sqrt(2) * z
                    p = 1.0
                    for e, i in zip(eta, idx):
                        q = F(e + s * u)
                        p *= q if Y[i] else 1 - q
                    acc += w * p
                tot -= math.log(max(acc / math.sqrt(math.pi), 1e-300))
            return tot

        start = [0.0] * (k + 1) + [0.0]
        n_units = len(units)
    else:
        raise ValueError('model must be "fe_logit", "re_logit" or "re_probit"')
    r = bfgs_minimize(nll, start, gtol=1e-8, max_iter=500)
    th = r["x"]
    # numerical Hessian of the negative log-likelihood for standard errors
    p = len(th)
    H = [[0.0] * p for _ in range(p)]
    for a in range(p):
        for c in range(a, p):
            ha, hc = 1e-4 * max(1.0, abs(th[a])), 1e-4 * max(1.0, abs(th[c]))

            def f(da, dc, *, a=a, c=c):
                t = list(th)
                t[a] += da
                t[c] += dc
                return nll(t)

            H[a][c] = H[c][a] = (f(ha, hc) - f(ha, -hc) - f(-ha, hc) + f(-ha, -hc)) / (4 * ha * hc)
    from ._qpcore import inverse

    try:
        Hi = inverse(H)
        se = [math.sqrt(Hi[i][i]) if Hi[i][i] > 0 else float("nan") for i in range(p)]
    except ZeroDivisionError:
        se = [float("nan")] * p
    out = {"loglik": -r["fun"], "n_units": n_units, "converged": r["converged"]}
    if model == "fe_logit":
        out.update(coef=th, se=se)
    else:
        out.update(coef=th[: k + 1], se=se[: k + 1], sigma=math.exp(th[k + 1]))
    return RichResult(
        title="Panel binary choice", summary_lines=[("model", model), ("loglik", out["loglik"])], payload=out
    )


def cheatsheet():
    return "panchc: conditional (FE) logit and random-effects logit/probit by Gauss-Hermite quadrature"
