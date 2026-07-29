# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Semi-supervised learning: small labeled set plus large unlabeled pool."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_semisupervised"]


def geron_semisupervised(X_l, y_l, X_u, alpha=1.0, gamma=1.0, fit_intercept=True):
    """
    Semi-supervised learning: small labeled set plus large unlabeled pool.

    Formula: minimize L_sup(f,L) + alpha * L_unsup(f,U)

    Uses the classical Laplacian-regularised least squares form of that
    objective, which has a closed-form solution instead of a heuristic
    self-training loop:

    ``L_sup = ||X_l theta - y_l||^2`` and
    ``L_unsup = f^T L f = sum_ij W_ij (f_i - f_j)^2 / 2`` with
    ``f = X_u theta``,

    so the unlabeled term is a *smoothness* penalty: predictions must
    agree between neighbouring unlabeled points (the cluster assumption).
    The graph ``W`` is the RBF affinity on the unlabeled pool and
    ``L = D - W``. The solution is
    ``theta = (X_l^T X_l + alpha X_u^T L X_u)^-1 X_l^T y_l``, so
    ``alpha = 0`` reduces exactly to supervised least squares and the
    roughness ``f^T L f`` falls monotonically as alpha grows.

    Parameters
    ----------
    X_l : array-like
        Labeled inputs (n_l, d).
    y_l : array-like
        Labels, length n_l.
    X_u : array-like
        Unlabeled pool (n_u, d), same width.
    alpha : float, default 1.0
        Weight on the unlabeled smoothness term (>= 0).
    gamma : float, default 1.0
        RBF width for the affinity graph (> 0).
    fit_intercept : bool, default True
        Prepend an intercept column (never penalised by the graph term).

    Returns
    -------
    result : RichResult
        Keys: theta, fitted, unlabeled_pred, sup_loss, roughness,
        objective, estimate, n, method.

    Examples
    --------
    With alpha = 0 the fit is exactly supervised least squares:

    >>> Xl = [[0.0], [1.0], [2.0]]
    >>> yl = [1.0, 2.0, 3.0]
    >>> Xu = [[0.5], [1.5], [3.0]]
    >>> r0 = geron_semisupervised(Xl, yl, Xu, alpha=0.0)
    >>> [round(float(v), 9) for v in r0["theta"]]
    [1.0, 1.0]
    >>> round(float(r0["sup_loss"]), 12)
    0.0

    Turning the unlabeled term up smooths the predictions over the pool:

    >>> r1 = geron_semisupervised(Xl, yl, Xu, alpha=5.0)
    >>> bool(r1["roughness"] < r0["roughness"])
    True
    >>> bool(r1["sup_loss"] > r0["sup_loss"])
    True

    References
    ----------
    Géron Ch 1
    """
    L1 = np.asarray(X_l, dtype=float)
    if L1.ndim == 1:
        L1 = L1.reshape(-1, 1)
    U = np.asarray(X_u, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if L1.ndim != 2 or L1.size == 0:
        raise ValueError("geron_semisupervised: X_l must be a non-empty (n_l, d) matrix")
    if U.ndim != 2 or U.size == 0:
        raise ValueError("geron_semisupervised: X_u must be a non-empty (n_u, d) matrix")
    if L1.shape[1] != U.shape[1]:
        raise ValueError(
            f"geron_semisupervised: X_l has {L1.shape[1]} features but X_u has {U.shape[1]}; "
            "both sets must live in the same space"
        )
    t = np.asarray(y_l, dtype=float).ravel()
    if t.size != L1.shape[0]:
        raise ValueError(f"geron_semisupervised: X_l has {L1.shape[0]} rows but y_l has {t.size} labels")
    if not (np.all(np.isfinite(L1)) and np.all(np.isfinite(U)) and np.all(np.isfinite(t))):
        raise ValueError("geron_semisupervised: inputs must be finite")
    a = float(alpha)
    if not np.isfinite(a) or a < 0:
        raise ValueError(f"geron_semisupervised: alpha must be non-negative and finite, got {a}")
    g = float(gamma)
    if not np.isfinite(g) or g <= 0:
        raise ValueError(f"geron_semisupervised: gamma must be positive and finite, got {g}")

    Dl = np.hstack([np.ones((L1.shape[0], 1)), L1]) if fit_intercept else L1
    Du = np.hstack([np.ones((U.shape[0], 1)), U]) if fit_intercept else U

    diff = U[:, None, :] - U[None, :, :]
    W = np.exp(-g * np.sum(diff * diff, axis=2))
    np.fill_diagonal(W, 0.0)
    Lap = np.diag(W.sum(axis=1)) - W

    M = Dl.T @ Dl + a * (Du.T @ Lap @ Du)
    theta = np.linalg.pinv(M) @ (Dl.T @ t)
    fitted = Dl @ theta
    f_u = Du @ theta
    sup = float(np.mean((fitted - t) ** 2))
    rough = float(f_u @ Lap @ f_u)

    return RichResult(
        title="Semi-supervised (Laplacian-regularised) fit",
        summary_lines=[
            ("Labeled", int(L1.shape[0])),
            ("Unlabeled", int(U.shape[0])),
            ("alpha", a),
            ("Supervised MSE", sup),
            ("Unlabeled roughness", rough),
        ],
        interpretation=(
            "The unlabeled pool never supplies a target; it supplies the geometry. The penalty says "
            "'points that sit together should be predicted alike', which is the cluster assumption made explicit."
        ),
        payload={
            "theta": theta,
            "fitted": fitted,
            "unlabeled_pred": f_u,
            "sup_loss": sup,
            "roughness": rough,
            "objective": float(np.sum((fitted - t) ** 2) + a * rough),
            "laplacian": Lap,
            "affinity": W,
            "alpha": a,
            "estimate": sup,
            "n": int(L1.shape[0] + U.shape[0]),
            "method": "Laplacian-regularised least squares: supervised loss + alpha * graph smoothness on the unlabeled pool",
        },
    )


def cheatsheet():
    return "hmsem: Semi-supervised learning: small labeled set plus large unlabeled pool"
