# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Supervised learning paradigm: learn mapping f(x)->y from labeled examples."""

from . import _array_core as np

from ._richresult import RichResult
from .hmbat import geron_batch_learning

__all__ = ["geron_supervised_learning"]


def geron_supervised_learning(X, y, ridge=0.0, fit_intercept=True):
    """
    Supervised learning paradigm: learn mapping f(x)->y from labeled examples.

    Formula: minimize L(f(X), Y) over f in hypothesis class

    Empirical risk minimisation over the linear hypothesis class. The fit
    itself is delegated to :func:`morie.fn.hmbat.geron_batch_learning`
    (closed-form ridge/least squares) rather than reimplemented; what this
    function adds is the part of the paradigm that matters and that a
    training-error number hides -- **leave-one-out risk**, computed
    exactly from the hat matrix,

    ``e_loo_i = e_i / (1 - h_ii)``,

    so the optimism of the empirical risk is measured rather than
    assumed. A perfectly interpolating model (``h_ii = 1``) has no
    leave-one-out estimate at all, and that is reported as an error
    instead of an infinity.

    Parameters
    ----------
    X : array-like
        Labeled inputs (n, d).
    y : array-like
        Targets, length n.
    ridge : float, default 0.0
        L2 penalty passed to the batch learner (>= 0).
    fit_intercept : bool, default True
        Include an intercept column.

    Returns
    -------
    result : RichResult
        Keys: theta, predict, fitted, empirical_risk, loo_risk, optimism,
        r2, estimate, n, method.

    Examples
    --------
    An exactly linear relation is learned exactly, and the leave-one-out
    risk agrees with the training risk because there is nothing to overfit:

    >>> X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    >>> r = geron_supervised_learning(X, [3.0, 5.0, 7.0, 9.0, 11.0])
    >>> [round(float(v), 9) for v in r["theta"]]
    [1.0, 2.0]
    >>> round(float(r["empirical_risk"]), 12)
    0.0
    >>> round(float(r["loo_risk"]), 12)
    0.0

    Noise the target and the empirical risk becomes optimistic -- the
    leave-one-out risk is larger:

    >>> r2 = geron_supervised_learning(X, [3.0, 5.5, 6.5, 9.0, 11.5])
    >>> bool(r2["loo_risk"] > r2["empirical_risk"])
    True
    >>> bool(r2["optimism"] > 0)
    True

    References
    ----------
    Géron Ch 1
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("geron_supervised_learning: X must be a non-empty (n, d) matrix of labeled inputs")
    t = np.asarray(y, dtype=float).ravel()
    if t.size != A.shape[0]:
        raise ValueError(
            f"geron_supervised_learning: {A.shape[0]} inputs but {t.size} labels; supervised learning needs a label per example"
        )
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(t))):
        raise ValueError("geron_supervised_learning: X and y must be finite")
    lam = float(ridge)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError(f"geron_supervised_learning: ridge must be non-negative and finite, got {lam}")

    inner = geron_batch_learning(A, t, fit_intercept=bool(fit_intercept), ridge=lam)
    theta = np.asarray(inner["theta"], dtype=float)
    resid = np.asarray(inner["residuals"], dtype=float)
    risk = float(inner["train_mse"])

    D = np.hstack([np.ones((A.shape[0], 1)), A]) if fit_intercept else A
    P = D.T @ D + lam * np.eye(D.shape[1])
    H = D @ np.linalg.pinv(P) @ D.T
    h = np.clip(np.diag(H), 0.0, 1.0)
    if np.any(h >= 1.0 - 1e-12):
        raise ValueError(
            "geron_supervised_learning: the model interpolates at least one point (leverage 1), so the "
            "leave-one-out risk is undefined; add data or increase `ridge`"
        )
    loo = float(np.mean((resid / (1.0 - h)) ** 2))

    return RichResult(
        title="Supervised learning (empirical risk minimisation)",
        summary_lines=[
            ("Examples", int(t.size)),
            ("Parameters", int(theta.size)),
            ("Empirical risk (MSE)", risk),
            ("Leave-one-out risk", loo),
        ],
        interpretation=(
            "Supervised learning minimises risk on the sample it can see; the gap to the leave-one-out "
            "risk is the optimism you pay for having fitted those same points."
        ),
        payload={
            "theta": theta,
            "predict": inner["predict"],
            "fitted": np.asarray(inner["fitted"], dtype=float),
            "residuals": resid,
            "empirical_risk": risk,
            "loo_risk": loo,
            "optimism": float(loo - risk),
            "leverage": h,
            "r2": float(inner["r2"]),
            "estimate": risk,
            "n": int(t.size),
            "method": "Empirical risk minimisation over linear hypotheses (fit via hmbat) with exact leave-one-out risk",
        },
    )


def cheatsheet():
    return "hmsup: Supervised learning paradigm: learn mapping f(x)->y from labeled examples"
