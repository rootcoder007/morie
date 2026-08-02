# morie.fn -- function file (rootcoder007/morie)
"""Marginal structural model estimated with instrumental variables."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["msm_iv"]


def msm_iv(y, treatment_history, instruments, covariate_history=None):
    r"""Cumulative-treatment MSM fitted by 2SLS on the instruments.

    When time-varying confounding is *unmeasured*, IPTW cannot fix it,
    but instruments can: the structural model

    .. math:: E[Y(\bar a)] = \beta_0 + \beta_1 \textstyle\sum_t a_t

    is fitted by two-stage least squares, instrumenting cumulative
    treatment with cumulative instrument (plus any measured
    covariates as included exogenous regressors). The reported
    first-stage F statistic is the weak-instrument diagnostic that
    decides whether the estimate means anything.

    Parameters
    ----------
    y : array-like, shape (n,)
        End-of-follow-up outcome.
    treatment_history : array-like, shape (n, T) or (n,)
        Treatment per period.
    instruments : array-like, shape (n, T) or (n,) or (n, q)
        Instruments; summed across columns when shaped like the
        treatment history.
    covariate_history : array-like, optional
        Measured covariates entering both stages.

    Returns
    -------
    RichResult
        keys: ``estimate`` (per-period effect), ``intercept``,
        ``first_stage_f``, ``weak_instrument`` (True when F < 10),
        ``ols_estimate`` (for contrast), ``n``, ``n_periods``,
        ``method``.

    References
    ----------
    Robins, J. M. (1994). Correcting for non-compliance in randomized
    trials using structural nested mean models. *Communications in
    Statistics - Theory and Methods*, 23(8), 2379-2412.

    Staiger, D. & Stock, J. H. (1997). Instrumental variables
    regression with weak instruments. *Econometrica*, 65(3), 557-586.
    (the first-stage F rule of thumb)
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(treatment_history, dtype=float)
    Zin = np.asarray(instruments, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if Zin.ndim == 1:
        Zin = Zin[:, None]
    n, T = A.shape
    if y.size != n or Zin.shape[0] != n:
        raise ValueError("y, treatment_history, instruments must share their first dimension.")
    if covariate_history is None:
        C = np.empty((n, 0))
    else:
        C = np.asarray(covariate_history, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"covariate_history has {C.shape[0]} rows but y has {n}.")
    if n < C.shape[1] + Zin.shape[1] + 4:
        raise ValueError("too few observations for the two stages.")

    cumA = A.sum(axis=1)
    one = np.ones(n)
    D1 = np.column_stack([one, Zin, C])
    b1, *_ = np.linalg.lstsq(D1, cumA, rcond=None)
    fit = D1 @ b1
    rss_u = float(((cumA - fit) ** 2).sum())
    D0 = np.column_stack([one, C])
    b0, *_ = np.linalg.lstsq(D0, cumA, rcond=None)
    rss_r = float(((cumA - D0 @ b0) ** 2).sum())
    q = Zin.shape[1]
    dof = n - D1.shape[1]
    F = ((rss_r - rss_u) / q) / (rss_u / dof) if rss_u > 0 and dof > 0 else float("inf")

    D2 = np.column_stack([one, fit, C])
    b2, *_ = np.linalg.lstsq(D2, y, rcond=None)
    Dols = np.column_stack([one, cumA, C])
    bols, *_ = np.linalg.lstsq(Dols, y, rcond=None)

    return RichResult(
        payload={
            "estimate": float(b2[1]),
            "intercept": float(b2[0]),
            "first_stage_f": float(F),
            "weak_instrument": bool(F < 10),
            "ols_estimate": float(bols[1]),
            "n": int(n),
            "n_periods": int(T),
            "method": "MSM on cumulative treatment fitted by 2SLS",
        }
    )


def cheatsheet():
    return "msmiv2: 2SLS of Y on cumulative treatment instrumented by Z; F < 10 = weak"
