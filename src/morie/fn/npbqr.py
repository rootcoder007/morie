# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric Bayes quantile regression."""

import math

from . import _horowitz as hrz
from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["np_bayes_quant_reg"]


def np_bayes_quant_reg(y, X, tau=0.5, alpha=1.0, niter=40, eps=1e-3):
    """Quantile regression with an asymmetric-Laplace likelihood and a
    Dirichlet-process mixture on the error scale.

    Kottas & Gelfand's point is that fixing the error distribution to a
    single asymmetric Laplace is stronger than the quantile restriction
    actually requires; the median (more generally the tau-quantile)
    restriction is preserved by any scale mixture, so a Dirichlet
    process is placed on the mixing distribution of the scale.  The
    location parameter is unaffected, which is why the regression
    coefficient can be read off the check-loss fit:

        b_hat = argmin_b sum_i rho_tau(y_i - x_i'b),
        rho_tau(u) = u (tau - 1{u < 0}),

    and only the error model changes.  The fit uses the shared
    fixed-iteration IRLS helper (``qirls``), which takes exactly the
    same path in both language arms.

    The scale part is reported in closed form: the ALD maximum
    likelihood scale is ``sigma_hat = (1/n) sum_i rho_tau(r_i)``, and
    the DP mixing distribution induces
    ``E[K_n] = sum_{i=1}^n alpha / (alpha + i - 1)`` distinct scales a
    priori, the same expression that governs any DP partition.

    Determinism: fixed IRLS iteration count, no sampling.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p)
        Design WITHOUT an intercept column; one is added.
    tau : float, default 0.5
        Quantile in (0, 1).  0.5 is the median regression of the paper.
    alpha : float, default 1.0
        DP concentration for the scale mixture, positive.
    niter : int, default 40
        IRLS iterations.
    eps : float, default 1e-3
        Residual floor in the IRLS weight.

    Returns
    -------
    RichResult
        ``beta``, ``estimate`` (the intercept), ``sigma`` (ALD scale),
        ``check_loss``, ``e_k`` (prior expected number of distinct
        scales), ``tau``, ``alpha``, ``n``, ``p``.

    References
    ----------
    Kottas, A. & Gelfand, A. E. (2001).  Bayesian semiparametric median
    regression modeling.  Journal of the American Statistical
    Association, 96(456), 1458--1468.
    doi:10.1198/016214501753382363
    """
    yv = C.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("np_bayes_quant_reg: y is empty")
    t = float(tau)
    if not (0.0 < t < 1.0):
        raise ValueError("np_bayes_quant_reg: tau must lie in (0, 1)")
    a = float(alpha)
    if a <= 0.0:
        raise ValueError("np_bayes_quant_reg: alpha must be positive")
    Xm = C.cbind1(C.mat(X))
    if len(Xm) != n:
        raise ValueError("np_bayes_quant_reg: X and y have different lengths")
    p = len(Xm[0])
    beta = [float(v) for v in hrz.qirls(Xm, yv, [1.0] * n, t,
                                        niter=int(niter), eps=float(eps))]
    loss = 0.0
    for i in range(n):
        r = yv[i] - sum(Xm[i][k] * beta[k] for k in range(p))
        loss += r * (t - (1.0 if r < 0.0 else 0.0))
    ek = 0.0
    for i in range(1, n + 1):
        ek += a / (a + i - 1.0)
    return RichResult(payload={
        "beta": beta, "estimate": beta[0], "sigma": loss / n,
        "check_loss": loss, "e_k": ek,
        "e_k_digamma": a * (core.digamma(a + n) - core.digamma(a)),
        "tau": t, "alpha": a, "n": n, "p": p,
        "method": "Quantile regression with a DP scale mixture (Kottas-Gelfand 2001)"})


def cheatsheet():
    return "npbqr: Quantile regression with a Dirichlet-process scale mixture"


npbayesquantreg = np_bayes_quant_reg
