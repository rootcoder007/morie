# morie.fn -- function file (rootcoder007/morie)
"""DR-DiD with propensity overlap trimming."""

import numpy as np

from ._richresult import RichResult
from .aiptdd import _logit_fit, aipw_did

__all__ = ["dr_did_overlap_trim"]


def dr_did_overlap_trim(y_pre, y_post, D, X, eps=0.1):
    r"""Doubly robust DiD on the overlap-trimmed subsample.

    Estimates the propensity :math:`\hat e(X)`, keeps only units with
    :math:`\hat e(X) \in [\varepsilon, 1-\varepsilon]`, and runs the
    Sant'Anna-Zhao doubly robust DiD on the trimmed sample. Crump et
    al. show discarding extreme-propensity units bounds the variance
    of the reweighted estimator; their rule of thumb is
    :math:`\varepsilon = 0.1`.

    Parameters
    ----------
    y_pre, y_post : array-like, shape (n,)
        Panel outcomes in the pre and post period.
    D : array-like of {0, 1}, shape (n,)
        Treatment group.
    X : array-like, shape (n,) or (n, p)
        Covariates for the propensity model.
    eps : float, default 0.1
        Overlap threshold.

    Returns
    -------
    RichResult
        keys: ``att``, ``se``, ``ci_low``, ``ci_high``, ``n_kept``, ``n_trimmed``,
        ``eps``, ``n``, ``method``.

    References
    ----------
    Crump, R. K., Hotz, V. J., Imbens, G. W. & Mitnik, O. A. (2009).
    Dealing with limited overlap in estimation of average treatment
    effects. *Biometrika*, 96(1), 187-199. (the 0.1 rule of thumb)

    Sant'Anna, P. H. C. & Zhao, J. (2020). Doubly robust
    difference-in-differences estimators. *Journal of Econometrics*,
    219(1), 101-122. (the estimator run on the kept sample)
    """
    y_pre = np.asarray(y_pre, dtype=float).ravel()
    y_post = np.asarray(y_post, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = y_pre.size
    if not (y_post.size == n and D.size == n and X.shape[0] == n):
        raise ValueError("y_pre, y_post, D, X must share their first dimension.")
    eps = float(eps)
    if not 0 < eps < 0.5:
        raise ValueError(f"eps must lie in (0, 0.5), got {eps}.")
    if not np.all(np.isin(D, (0.0, 1.0))):
        raise ValueError("D must be binary 0/1.")

    e = _logit_fit(X, D)
    keep = (e >= eps) & (e <= 1 - eps)
    if keep.sum() < 8 or D[keep].min() == D[keep].max():
        raise ValueError("overlap trimming left too few units in one arm; lower eps.")

    sub = aipw_did(y_pre[keep], y_post[keep], D[keep], X[keep])
    return RichResult(
        payload={
            "att": sub["att"],
            "se": sub["se"],
            "ci_low": sub["ci_low"],
            "ci_high": sub["ci_high"],
            "n_kept": int(keep.sum()),
            "n_trimmed": int(n - keep.sum()),
            "eps": eps,
            "n": int(n),
            "method": "DR-DiD with propensity overlap trimming (Crump 0.1 rule)",
        }
    )


def cheatsheet():
    return "drovrl: trim e(X) outside [eps, 1-eps], then Sant'Anna-Zhao DR-DiD"
