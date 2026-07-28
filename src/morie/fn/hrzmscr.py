# morie.fn -- function file (rootcoder007/morie)
"""Maximum score estimator."""

import numpy as np

from ._horowitz import silverman_bw
from ._richresult import RichResult

__all__ = ["hrz_maximum_score"]


from scipy import optimize


def hrz_maximum_score(X, y, beta0=None, n_restarts=8, seed=0):
    r"""Manski's maximum score estimator (Horowitz Ch. 3):

    .. math:: \hat\beta = \arg\max_{b:\,|b_1|=1}
              \sum_i (2Y_i - 1)\,\mathbf 1\{X_i'b > 0\}.

    Requires only a conditional-MEDIAN restriction on the error, so it
    tolerates arbitrary heteroskedasticity of unknown form -- far
    weaker than probit's distributional assumption. The price is a
    discontinuous objective: the rate is :math:`n^{-1/3}` with a
    non-normal (Chernoff) limit, so ordinary standard errors do NOT
    apply, and that is stated rather than a spurious SE returned.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Covariates.
    y : array-like of {0, 1}, shape (n,)
        Binary response.
    beta0 : array-like, optional
        Starting value.
    n_restarts : int, default 8
        Random restarts, since the objective is a step function.
    seed : int, default 0
        RNG seed for restarts.

    Returns
    -------
    RichResult
        keys: ``beta``, ``score``, ``rate_exponent`` (-1/3),
        ``limit_distribution`` ("Chernoff, non-normal"),
        ``standard_errors_valid`` (False), ``n``, ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 3 (the maximum score estimator).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.size:
        X = X.T
    if X.shape[0] != y.size:
        raise ValueError("X must have one row per entry of y.")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y must be binary 0/1.")
    n, d = X.shape
    if d < 2:
        raise ValueError("need at least 2 covariates.")
    s = 2.0 * y - 1.0

    def neg(rest):
        b = np.r_[1.0, rest]
        return -float(np.sum(s * (X @ b > 0)))

    rng = np.random.default_rng(seed)
    best, best_val = None, np.inf
    starts = [np.zeros(d - 1) if beta0 is None else
              np.atleast_1d(np.asarray(beta0, dtype=float))[1:]]
    starts += [rng.standard_normal(d - 1) for _ in range(int(n_restarts))]
    for st in starts:
        r = optimize.minimize(neg, st, method="Nelder-Mead",
                              options={"maxiter": 3000, "fatol": 1e-9})
        if r.fun < best_val:
            best_val, best = r.fun, r.x
    return RichResult(payload={"beta": np.r_[1.0, best], "score": -best_val,
                               "rate_exponent": -1.0 / 3.0,
                               "limit_distribution": "Chernoff, non-normal",
                               "standard_errors_valid": False,
                               "n": int(n), "d": int(d),
                               "method": "Manski max score; median restriction only, n^{-1/3}"})


def cheatsheet():
    return "hrzmscr: n^{-1/3} Chernoff limit -- ordinary SEs do NOT apply"
