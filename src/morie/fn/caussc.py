# morie.fn -- function file (rootcoder007/morie)
"""Abadie-Diamond-Hainmueller synthetic control weights."""

from . import _array_core as np
from ._sci_core import optimize

from ._richresult import RichResult

__all__ = ["causal_synthetic_control"]


def causal_synthetic_control(X1_pre, X0_pre, V=None):
    r"""Simplex-constrained synthetic control weights.

    Solves the Abadie-Diamond-Hainmueller programme

    .. math:: \min_w (X_1 - X_0 w)' V (X_1 - X_0 w)
              \quad \text{s.t.} \quad w_j \ge 0,\; \sum_j w_j = 1,

    via nonnegative least squares on the :math:`V^{1/2}`-scaled system
    augmented with a heavily weighted sum-to-one row (the standard NNLS
    reduction of the simplex-constrained QP).

    Parameters
    ----------
    X1_pre : array-like, shape (k,)
        Treated unit's pre-treatment predictor vector.
    X0_pre : array-like, shape (k, J)
        Donor pool predictor matrix, one column per donor.
    V : array-like, optional
        Predictor weights: a length-k vector (diagonal) or (k, k) PSD
        matrix. Default: identity.

    Returns
    -------
    RichResult
        keys: ``weights`` (J,), ``rmse_pre``, ``n_donors``, ``method``.

    References
    ----------
    Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic
    control methods for comparative case studies: estimating the
    effect of California's tobacco control program. *Journal of the
    American Statistical Association*, 105(490), 493-505.
    """
    x1 = np.asarray(X1_pre, dtype=float).ravel()
    X0 = np.asarray(X0_pre, dtype=float)
    if X0.ndim != 2:
        raise ValueError("X0_pre must be 2-D (k predictors x J donors).")
    k, J = X0.shape
    if x1.size != k:
        raise ValueError(f"X1_pre has {x1.size} predictors but X0_pre has {k} rows.")
    if J < 2:
        raise ValueError("need at least 2 donors.")

    if V is None:
        Vs = np.eye(k)
    else:
        V = np.asarray(V, dtype=float)
        if V.ndim == 1:
            if V.size != k or np.any(V < 0):
                raise ValueError("diagonal V must be length k and nonnegative.")
            Vs = np.diag(np.sqrt(V))
        else:
            if V.shape != (k, k):
                raise ValueError(f"V must be (k, k) = ({k}, {k}).")
            vals, vecs = np.linalg.eigh((V + V.T) / 2)
            if vals.min() < -1e-8:
                raise ValueError("V must be positive semidefinite.")
            Vs = vecs @ np.diag(np.sqrt(np.maximum(vals, 0))) @ vecs.T

    scale = max(np.abs(Vs @ X0).max(), 1.0)
    K = 1e4 * scale  # ponytail: big-row trick enforces sum(w)=1 to ~1e-8
    A = np.vstack([Vs @ X0, np.full((1, J), K)])
    b = np.concatenate([Vs @ x1, [K]])
    w, _ = optimize.nnls(A, b)
    w = w / w.sum()

    gap = x1 - X0 @ w
    rmse = float(np.sqrt(np.mean((Vs @ gap) ** 2)))

    return RichResult(
        payload={
            "weights": w,
            "rmse_pre": rmse,
            "n_donors": int(J),
            "method": "Abadie-Diamond-Hainmueller synthetic control weights",
        }
    )


def cheatsheet():
    return "caussc: min (X1-X0w)'V(X1-X0w) on the simplex via NNLS big-row trick"
