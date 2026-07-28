# morie.fn -- function file (rootcoder007/morie)
"""Instrument validity diagnostics."""

import numpy as np

from ._richresult import RichResult

__all__ = ["hrz_instrument_check"]


def hrz_instrument_check(X, Z, U=None, y=None):
    r"""Diagnostics for the instrument conditions (Horowitz Ch. 6):

    identification needs :math:`E[U|Z] = 0` (exogeneity) together with
    variation in X that Z explains (relevance).

    Both are reported and they fail in different ways: a weak but
    valid instrument gives large variance, while a strong but invalid
    one gives confident nonsense. Exogeneity is NOT testable without
    further restrictions -- the returned correlation is a diagnostic
    against a supplied residual, not a test, and the key is named
    accordingly.

    Parameters
    ----------
    X : array-like, shape (n,) or (n, d)
        Endogenous regressor(s).
    Z : array-like, shape (n,) or (n, q)
        Instrument(s).
    U : array-like, optional
        Residuals, for the exogeneity diagnostic.
    y : array-like, optional
        Response; residuals are formed by 2SLS when U is omitted.

    Returns
    -------
    RichResult
        keys: ``first_stage_r2``, ``first_stage_F``, ``relevant``,
        ``corr_U_Z`` (diagnostic only), ``exogeneity_testable``
        (False), ``n``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 6 (instrumental variables).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if X.shape[0] < X.shape[1]:
        X = X.T
    if Z.shape[0] < Z.shape[1]:
        Z = Z.T
    n = X.shape[0]
    if Z.shape[0] != n:
        raise ValueError("X and Z must have the same number of rows.")
    Zc = np.column_stack([np.ones(n), Z])
    q = Z.shape[1]
    x1 = X[:, 0]
    coef, *_ = np.linalg.lstsq(Zc, x1, rcond=None)
    fit = Zc @ coef
    ss_res = float(np.sum((x1 - fit) ** 2))
    ss_tot = float(np.sum((x1 - x1.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = max(n - q - 1, 1)
    F = (r2 / q) / ((1 - r2) / dof) if r2 < 1 else np.inf

    corr = None
    if U is not None:
        u = np.asarray(U, dtype=float).ravel()
        if u.size != n:
            raise ValueError("U must have one entry per row of X.")
        corr = float(np.corrcoef(u, Z[:, 0])[0, 1])
    elif y is not None:
        yy = np.asarray(y, dtype=float).ravel()
        if yy.size != n:
            raise ValueError("y must have one entry per row of X.")
        b2, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), fit]), yy, rcond=None)
        u = yy - np.column_stack([np.ones(n), x1]) @ b2
        corr = float(np.corrcoef(u, Z[:, 0])[0, 1])
    return RichResult(payload={"first_stage_r2": float(r2),
                               "first_stage_F": float(F),
                               "relevant": bool(F > 10.0),  # the usual rule of thumb
                               "corr_U_Z": corr,
                               "exogeneity_testable": False,
                               "n": int(n), "n_instruments": int(q),
                               "method": "Relevance is testable; exogeneity is NOT, and is not claimed"})


def cheatsheet():
    return "hrzinst: weak-but-valid inflates variance; strong-but-invalid gives confident nonsense"
