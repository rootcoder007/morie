# morie.fn -- function file (rootcoder007/morie)
"""Cox partial likelihood."""

import numpy as np

from ._kosorok import cox_score
from ._richresult import RichResult

__all__ = ["kosorok_ch3_cox_partial_likelihood"]


def kosorok_ch3_cox_partial_likelihood(beta, Z, V, d, n=None):
    r"""Cox partial likelihood, the criterion whose maximiser is an
    efficient estimator of beta (Kosorok Ch. 3, eq. 3.4, p. 43):

    .. math:: \tilde L_n(\beta) = \prod_{i=1}^n \left[
              \frac{e^{\beta' Z_i}}
                   {\sum_{j:\,V_j \ge V_i} e^{\beta' Z_j}}
              \right]^{d_i}.

    Each factor compares the subject who failed against everyone still
    at risk at that moment, and the baseline hazard cancels out of the
    ratio entirely -- which is why beta can be estimated efficiently
    without ever estimating Lambda.

    The standard error comes from the observed information (the
    inverse of which is the efficient variance), not from the spread
    of the coefficient vector.

    Parameters
    ----------
    beta : array-like, shape (p,)
        Coefficients.
    Z : array-like, shape (n, p)
        Covariates.
    V : array-like, shape (n,)
        Observed times (min of event and censoring).
    d : array-like of {0, 1}, shape (n,)
        Event indicators.
    n : int, optional
        Sample size; taken from the data.

    Returns
    -------
    RichResult
        keys: ``loglik``, ``partial_likelihood``, ``estimate`` (the
        maximiser), ``se``, ``ci_lower``, ``ci_upper``, ``score``,
        ``information``, ``n_events``, ``n``, ``method``.

    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 3, eq. (3.4), p. 43.

    Cox, D. R. (1972). Regression models and life-tables. *Journal of
    the Royal Statistical Society B*, 34(2), 187-202.
    """
    from scipy import optimize

    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    V = np.asarray(V, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    nobs, p = Z.shape
    if V.size != nobs or d.size != nobs:
        raise ValueError("V and d must have one entry per row of Z.")
    if not np.all(np.isin(d, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")
    if n is not None and int(n) != nobs:
        raise ValueError(f"n = {n} does not match the {nobs} rows of Z.")
    if d.sum() == 0:
        raise ValueError("no events; the partial likelihood is degenerate.")
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    if beta.size != p:
        raise ValueError(f"beta must have {p} entries, got {beta.size}.")

    at_beta = cox_score(beta, Z, V, d)
    res = optimize.root(lambda b: cox_score(b, Z, V, d)["score"], np.zeros(p),
                        method="hybr")
    hat = res.x
    at_hat = cox_score(hat, Z, V, d)
    info = at_hat["information"]
    try:
        cov = np.linalg.inv(info)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    return RichResult(
        payload={
            "loglik": at_beta["loglik"],
            "partial_likelihood": float(np.exp(at_beta["loglik"])),
            "estimate": hat if p > 1 else float(hat[0]),
            "se": se if p > 1 else float(se[0]),
            "ci_lower": hat - 1.96 * se if p > 1 else float(hat[0] - 1.96 * se[0]),
            "ci_upper": hat + 1.96 * se if p > 1 else float(hat[0] + 1.96 * se[0]),
            "score": at_beta["score"], "information": info,
            "converged": bool(res.success),
            "n_events": at_hat["n_events"], "n": int(nobs),
            "method": "Cox partial likelihood (Kosorok eq. 3.4); SE from the information",
        }
    )


def cheatsheet():
    return "ksr064: baseline hazard cancels in the ratio; SE from observed information"
