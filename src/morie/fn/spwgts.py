# morie.fn -- function file (rootcoder007/morie)
"""Spline-based propensity weights."""

import numpy as np

from ._richresult import RichResult
from .aiptdd import _logit_fit

__all__ = ["spline_weights"]


def _natural_spline_basis(x, knots):
    """Natural cubic spline basis (Hastie-Tibshirani-Friedman Eq. 5.4-5.5)."""
    knots = np.sort(np.asarray(knots, dtype=float))
    K = knots.size
    kK, kK1 = knots[-1], knots[-2]

    def d(j):
        return (np.maximum(x - knots[j], 0) ** 3 - np.maximum(x - kK, 0) ** 3) / (kK - knots[j])

    cols = [x] + [d(j) - d(K - 2) for j in range(K - 2)]
    return np.column_stack(cols)


def spline_weights(A, H, knots=None, trunc=0.01):
    r"""Inverse-probability weights from a spline propensity model.

    Fits the propensity of a binary treatment/exposure ``A`` by
    logistic regression on a natural cubic spline expansion of the
    scalar history/covariate ``H`` -- Westreich et al.'s point that
    misspecified linear propensity models propagate into the weights,
    and flexible basis expansions guard against it -- then returns the
    Horvitz-Thompson weights

    .. math:: w_i = \frac{A_i}{\hat e(H_i)}
              + \frac{1 - A_i}{1 - \hat e(H_i)}.

    Parameters
    ----------
    A : array-like of {0, 1}, shape (n,)
        Binary treatment.
    H : array-like, shape (n,) or (n, p)
        Covariate; the spline expansion is applied to each column.
    knots : array-like, optional
        Interior + boundary knot locations (at least 3). Default: the
        {10, 35, 65, 90} percentiles of each column.
    trunc : float, default 0.01
        Truncate fitted propensities to [trunc, 1 - trunc] before
        weighting (Cole & Hernan 2008): a handful of near-0/1 fitted
        scores otherwise dominate the weighted sample.

    Returns
    -------
    RichResult
        keys: ``weights``, ``propensity``, ``ess``, ``n``, ``method``.

    References
    ----------
    Westreich, D., Lessler, J. & Funk, M. J. (2010). Propensity score
    estimation: neural networks, support vector machines, decision
    trees (CART), and meta-classifiers as alternatives to logistic
    regression. *Journal of Clinical Epidemiology*, 63(8), 826-833.
    (flexible propensity estimation instead of a rigid linear logit)

    Cole, S. R. & Hernan, M. A. (2008). Constructing inverse
    probability weights for marginal structural models. *American
    Journal of Epidemiology*, 168(6), 656-664. doi:10.1093/aje/kwn164.

    Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
    Statistical Learning* (2nd ed.). Springer. Sec. 5.2.1, Eqs.
    (5.4)-(5.5) (natural cubic spline basis).
    """
    A = np.asarray(A, dtype=float).ravel()
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("A must be binary 0/1.")
    H = np.asarray(H, dtype=float)
    if H.ndim == 1:
        H = H[:, None]
    if H.shape[0] != A.size:
        raise ValueError(f"H has {H.shape[0]} rows but A has {A.size}.")
    if A.sum() == 0 or A.sum() == A.size:
        raise ValueError("need both treated and untreated units.")

    basis_cols = []
    for j in range(H.shape[1]):
        col = H[:, j]
        kj = np.percentile(col, [10, 35, 65, 90]) if knots is None else np.asarray(knots, dtype=float)
        if np.unique(kj).size < 3:
            raise ValueError("need at least 3 distinct knots.")
        basis_cols.append(_natural_spline_basis(col, np.unique(kj)))
    X = np.column_stack(basis_cols)

    if not 0 <= trunc < 0.5:
        raise ValueError(f"trunc must lie in [0, 0.5), got {trunc}.")
    e = np.clip(_logit_fit(X, A), max(trunc, 1e-6), 1 - max(trunc, 1e-6))
    w = A / e + (1 - A) / (1 - e)
    ess = float(w.sum() ** 2 / (w**2).sum())

    return RichResult(
        payload={
            "weights": w,
            "propensity": e,
            "ess": ess,
            "n": int(A.size),
            "method": "Spline-based propensity weights",
        }
    )


def cheatsheet():
    return "spwgts: IP weights from natural-cubic-spline logistic propensity"
