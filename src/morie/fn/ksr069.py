# morie.fn -- function file (rootcoder007/morie)
"""Cox likelihood with the Breslow baseline estimator."""

from . import _array_core as np

from ._kosorok import cox_score
from ._richresult import RichResult

__all__ = ["kosorok_ch3_cox_likelihood_breslow"]


def kosorok_ch3_cox_likelihood_breslow(beta, Lambda, Z, V, d, n=None):
    r"""Cox likelihood profiled over the baseline hazard, with the
    Breslow estimator (Kosorok Ch. 3):

    .. math:: \hat\Lambda(t) = \sum_{i:\,V_i \le t}
              \frac{d_i}{\sum_{j:\,V_j \ge V_i} e^{\beta' Z_j}}.

    Breslow's estimator IS the profile maximiser of the baseline
    hazard given beta, so plugging it back in reproduces the partial
    likelihood of :mod:`morie.fn.ksr064` up to a constant. That
    equivalence is the reason profiling out an infinite-dimensional
    nuisance costs nothing here, and it is asserted in the tests
    rather than left as folklore.

    Parameters
    ----------
    beta : array-like, shape (p,)
        Coefficients.
    Lambda : array-like or None
        A supplied baseline; when None the Breslow estimate is used.
        A supplied Lambda is used for the full likelihood only.
    Z : array-like, shape (n, p)
        Covariates.
    V : array-like, shape (n,)
        Observed times.
    d : array-like of {0, 1}, shape (n,)
        Event indicators.
    n : int, optional
        Sample size; taken from the data.

    Returns
    -------
    RichResult
        keys: ``breslow_times``, ``breslow_cumhaz``, ``loglik``,
        ``estimate``, ``se``, ``information``, ``n_events``, ``n``,
        ``method``.

    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 3.

    Breslow, N. E. (1972). Discussion of Professor Cox's paper.
    *Journal of the Royal Statistical Society B*, 34(2), 216-217.
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
        raise ValueError("no events; the baseline hazard is unidentified.")
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    if beta.size != p:
        raise ValueError(f"beta must have {p} entries, got {beta.size}.")

    w = np.exp(Z @ beta)
    ev = np.sort(V[d == 1])
    inc = np.array([1.0 / max(float(w[V >= t].sum()), 1e-300) for t in ev])
    cum = np.cumsum(inc)

    at_beta = cox_score(beta, Z, V, d)
    res = optimize.root(lambda b: cox_score(b, Z, V, d)["score"], np.zeros(p),
                        method="hybr")
    hat = res.x
    info = cox_score(hat, Z, V, d)["information"]
    try:
        se = np.sqrt(np.diag(np.linalg.inv(info)))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    return RichResult(
        payload={
            "breslow_times": ev, "breslow_cumhaz": cum,
            "loglik": at_beta["loglik"],
            "estimate": hat if p > 1 else float(hat[0]),
            "se": se if p > 1 else float(se[0]),
            "information": info, "converged": bool(res.success),
            "n_events": int(d.sum()), "n": int(nobs),
            "method": "Breslow baseline = the profile maximiser given beta",
        }
    )


def cheatsheet():
    return "ksr069: Breslow IS the profile maximiser; profiling costs nothing"
