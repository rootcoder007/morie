# morie.fn -- function file (rootcoder007/morie)
"""Smoothed maximum score."""

from . import _array_core as np

from ._horowitz import silverman_bw
from ._richresult import RichResult

__all__ = ["hrz_smoothed_max_score", "horowitz_smoothed_max_score"]


from scipy import optimize, stats as _st


def hrz_smoothed_max_score(X, y, h=None, beta0=None, r=2):
    r"""Horowitz's smoothed maximum score estimator (Horowitz Sec. 4.3.3):

    .. math:: \hat\beta = \arg\max_{b:\,|b_1|=1} \frac1n
              \sum_i (2Y_i - 1)\, K\!\left(\frac{X_i'b}{h_n}\right).

    Replacing Manski's indicator with a smooth kernel CDF makes the
    objective differentiable, which lifts the rate from
    :math:`n^{-1/3}` to :math:`n^{-r/(2r+1)}` and restores asymptotic
    NORMALITY -- so standard errors become meaningful again. With a
    smoothness order r = 2 the rate is :math:`n^{-2/5}`; higher-order
    kernels do better. This is the estimator the chapter is named for.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Covariates.
    y : array-like of {0, 1}
        Binary response.
    h : float, optional
        Smoothing bandwidth.
    beta0 : array-like, optional
        Starting value.
    r : int, default 2
        Assumed smoothness order, used for the reported rate.

    Returns
    -------
    RichResult
        keys: ``beta``, ``objective``, ``bandwidth``,
        ``rate_exponent``, ``limit_distribution`` ("normal"),
        ``standard_errors_valid`` (True), ``n``, ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 4.3.3 (estimating beta: the smoothed
    maximum-score estimator).
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
    r = int(r)
    if r < 1:
        raise ValueError(f"r must be at least 1, got {r}.")
    s = 2.0 * y - 1.0
    hh = float(n ** (-1.0 / (2 * r + 1))) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")

    def neg(rest, s1=1.0):
        b = np.r_[s1, rest]
        # smooth indicator: the kernel CDF, so the objective is C^1
        return -float(np.mean(s * _st.norm.cdf((X @ b) / hh)))

    start = np.zeros(d - 1) if beta0 is None else \
        np.atleast_1d(np.asarray(beta0, dtype=float))[1:]
    # |b_1| = 1 covers both signs; optimise each half and keep the better.
    res, s1 = None, 1.0
    for cand_s1 in (1.0, -1.0):
        cand = optimize.minimize(neg, start, args=(cand_s1,), method="BFGS")
        if res is None or cand.fun < res.fun:
            res, s1 = cand, cand_s1
    return RichResult(payload={"beta": np.r_[s1, res.x],
                               "objective": float(-res.fun), "bandwidth": hh,
                               "rate_exponent": -r / (2.0 * r + 1.0),
                               "limit_distribution": "normal",
                               "standard_errors_valid": True,
                               "converged": bool(res.success),
                               "n": int(n), "d": int(d),
                               "method": "Smoothed max score; normality restored, rate n^{-r/(2r+1)}"})


def cheatsheet():
    return "hrzsms: smoothing buys normality AND a faster rate than n^{-1/3}"


#: Catalogue alias for :func:`hrz_smoothed_max_score`.
horowitz_smoothed_max_score = hrz_smoothed_max_score
