# morie.fn -- function file (rootcoder007/morie)
"""Double ML for instrumental variables (partially linear IV)."""

import numpy as np

from ._richresult import RichResult
from .medML import _pred, _ridge

__all__ = ["causal_dml_iv"]


def causal_dml_iv(y, D, Z, X, n_folds=5, seed=0):
    r"""Cross-fitted IV estimate of a partially linear model.

    For :math:`Y = \theta D + g(X) + \varepsilon` with an instrument Z,
    Chernozhukov et al.'s partialling-out moment is

    .. math:: \hat\theta = \frac{\sum_i \tilde Z_i \tilde Y_i}
                                {\sum_i \tilde Z_i \tilde D_i},

    where each tilde is the residual after removing the cross-fitted
    prediction from X. Cross-fitting is what makes the estimator
    root-n consistent when the nuisance functions are estimated
    flexibly; the plug-in version is not.

    Parameters
    ----------
    y, D, Z : array-like, shape (n,)
        Outcome, treatment, instrument.
    X : array-like, shape (n, p)
        Controls entering the nuisance functions.
    n_folds : int, default 5
        Cross-fitting folds.
    seed : int, default 0
        Fold RNG seed.

    Returns
    -------
    RichResult
        keys: ``theta``, ``se``, ``first_stage`` (the residualised
        Z-D covariance, a weak-instrument diagnostic), ``n_folds``,
        ``n``, ``method``.

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
    Hansen, C., Newey, W. & Robins, J. (2018). Double/debiased machine
    learning for treatment and structural parameters. *The
    Econometrics Journal*, 21(1), C1-C68. Sec. 4.2 (partially linear
    IV).
    """
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = y.size
    if not (D.size == n and Z.size == n and X.shape[0] == n):
        raise ValueError("y, D, Z, X must share their first dimension.")
    k = int(n_folds)
    if not 2 <= k <= n // 2:
        raise ValueError(f"n_folds must lie in [2, {n // 2}], got {k}.")

    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % k
    ry, rd, rz = np.empty(n), np.empty(n), np.empty(n)
    for f in range(k):
        tr, te = folds != f, folds == f
        for src, dst in ((y, ry), (D, rd), (Z, rz)):
            b = _ridge(X[tr], src[tr])
            dst[te] = src[te] - _pred(b, X[te])

    denom = float(rz @ rd)
    if abs(denom) < 1e-10:
        raise ValueError("residualised instrument is orthogonal to treatment; theta unidentified.")
    theta = float(rz @ ry) / denom
    psi = rz * (ry - theta * rd)
    se = float(np.sqrt((psi**2).sum()) / abs(denom))

    return RichResult(
        payload={
            "theta": theta,
            "se": se,
            "first_stage": denom / n,
            "n_folds": k,
            "n": int(n),
            "method": "Double ML IV (partially linear, cross-fitted partialling-out)",
        }
    )


def cheatsheet():
    return "causdmliv: theta = <rz, ry> / <rz, rd> on cross-fitted residuals"
