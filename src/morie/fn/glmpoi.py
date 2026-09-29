# morie.fn -- function file (rootcoder007/morie)
"""Poisson regression with R-style verbose result."""

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult


def _sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""


def glmpoi(X, y, add_intercept: bool = True, tol: float = 1e-12, maxit: int = 100):
    r"""Poisson regression (GLM, log link) by iteratively reweighted least squares.

    ``log E[y] = X b``; IRLS with working weights ``mu`` and working response
    ``eta + (y - mu)/mu`` from ``mu = y + 0.1``, stopping when the relative
    change of the deviance is below ``tol`` (McCullagh and Nelder 1989, sec.
    2.5; ``stats::glm(family = poisson)``). Standard errors from
    ``(X'WX)^{-1}``, Wald z p-values, deviance ``2 sum [y log(y/mu) - (y -
    mu)]``, ``AIC = -2 ll + 2p`` and the Pearson dispersion ``sum (y - mu)^2 /
    mu / (n - p)`` (a warning above 1.5).

    References
    ----------
    McCullagh, P. and Nelder, J. A. (1989). *Generalized Linear Models*, 2nd
    ed. Chapman and Hall.

    Examples
    --------
    >>> r = glmpoi([[0.0], [1.0], [2.0], [3.0], [4.0]], [1, 2, 2, 5, 8])
    >>> [round(v, 8) for v in r["coef"]]
    [-0.03447284, 0.52657755]
    """
    Xm = [
        [float(v) for v in (r if hasattr(r, "__len__") else [r])] for r in (X.tolist() if hasattr(X, "tolist") else X)
    ]
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    if any(v < 0 for v in yv):
        raise ValueError("Poisson y must be non-negative.")
    if add_intercept:
        Xm = [[1.0] + r for r in Xm]
    n, p = len(Xm), len(Xm[0])

    def dev(mu):
        return 2.0 * ssum((yv[i] * math.log(yv[i] / mu[i]) if yv[i] > 0 else 0.0) - (yv[i] - mu[i]) for i in range(n))

    mu = [v + 0.1 for v in yv]
    eta = [math.log(v) for v in mu]
    d_old = dev(mu)
    for _ in range(maxit):
        z = [eta[i] + (yv[i] - mu[i]) / mu[i] for i in range(n)]
        A = inverse([[ssum(mu[i] * Xm[i][a] * Xm[i][b] for i in range(n)) for b in range(p)] for a in range(p)])
        g = [ssum(mu[i] * Xm[i][a] * z[i] for i in range(n)) for a in range(p)]
        beta = [ssum(A[a][c] * g[c] for c in range(p)) for a in range(p)]
        eta = [ssum(r[a] * beta[a] for a in range(p)) for r in Xm]
        mu = [math.exp(v) for v in eta]
        d_new = dev(mu)
        done = abs(d_new - d_old) / (abs(d_new) + 0.1) < tol
        d_old = d_new
        if done:
            break
    V = inverse([[ssum(mu[i] * Xm[i][a] * Xm[i][b] for i in range(n)) for b in range(p)] for a in range(p)])
    se = [math.sqrt(V[a][a]) for a in range(p)]
    pv = [math.erfc(abs(b / s) / math.sqrt(2.0)) for b, s in zip(beta, se)]
    ll = ssum(yv[i] * eta[i] - mu[i] - math.lgamma(yv[i] + 1.0) for i in range(n))
    pearson = ssum((yv[i] - mu[i]) ** 2 / mu[i] for i in range(n))
    disp = pearson / (n - p) if n > p else float("nan")
    rows = [
        ["(Intercept)" if i == 0 and add_intercept else f"x{i}", f"{b:.4g}", f"{s:.4g}", f"{q:.4g}", _sig(q)]
        for i, (b, s, q) in enumerate(zip(beta, se, pv))
    ]
    warnings = []
    if disp == disp and disp > 1.5:
        warnings.append(
            f"overdispersion: Pearson chi^2/df = {disp:.2f} > 1.5; Poisson assumes equidispersion. "
            "Consider negative binomial."
        )
    return RichResult(
        title="Poisson regression (GLM, log link)",
        summary_lines=[
            ("AIC", -2 * ll + 2 * p),
            ("Deviance", d_old),
            ("Pearson chi^2", pearson),
            ("Pearson chi^2 / df", disp),
            ("n observations", n),
            ("n parameters", p),
            ("Log-likelihood", ll),
        ],
        tables=[
            {
                "title": "Coefficients (link scale = log mean):",
                "headers": ["Variable", "Estimate", "Std. Error", "Pr(>|z|)", "Sig."],
                "rows": rows,
            }
        ],
        warnings=warnings,
        interpretation="Coefficients are on the log-mean scale; exp(coef) = rate ratio.",
        payload={
            "coef": beta,
            "se": se,
            "pvalues": pv,
            "aic": -2 * ll + 2 * p,
            "deviance": d_old,
            "loglik": ll,
            "fitted": mu,
            "overdispersion": disp,
        },
    )


def cheatsheet() -> str:
    return "glmpoi: glmpoi(X, y, add_intercept) -> Poisson GLM by IRLS (stats::glm family = poisson)."
