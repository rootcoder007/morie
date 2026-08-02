# morie.fn -- function file (rootcoder007/morie)
"""Synthetic control subset selection (LASSO-relaxed)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_synthetic_subset"]


def causal_synthetic_subset(X1_pre, X0_pre, lam=0.1, max_iter=5000, tol=1e-10):
    r"""Sparse donor selection by nonnegative LASSO, then simplex refit.

    On the simplex the :math:`\ell_1` norm is identically 1, so the
    penalty must act on the *relaxed* problem: solve

    .. math:: \min_{w \ge 0} \tfrac12 \|X_1 - X_0 w\|_2^2
              + \lambda \sum_j w_j

    by projected (proximal) gradient descent -- for nonnegative w the
    LASSO soft-threshold is just a shift of the gradient step -- and
    take the nonzero coordinates as the selected donor subset. The
    reported weights are renormalised to the simplex over that support,
    restoring the Abadie-style interpretation with a sparse donor pool.

    Parameters
    ----------
    X1_pre : array-like, shape (k,)
        Treated unit's pre-treatment vector.
    X0_pre : array-like, shape (k, J)
        Donor pool matrix.
    lam : float, default 0.1
        Sparsity penalty; larger values select fewer donors.
    max_iter, tol :
        Projected-gradient controls.

    Returns
    -------
    RichResult
        keys: ``weights`` (J, simplex over the support), ``support``
        (indices), ``n_selected``, ``lam``, ``rmse_pre``, ``method``.

    References
    ----------
    Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic
    control methods for comparative case studies. *JASA*, 105(490),
    493-505. (the simplex programme being sparsified)
    """
    x1 = np.asarray(X1_pre, dtype=float).ravel()
    X0 = np.asarray(X0_pre, dtype=float)
    if X0.ndim != 2 or X0.shape[0] != x1.size:
        raise ValueError("X0_pre must be (k, J) matching X1_pre.")
    lam = float(lam)
    if lam < 0:
        raise ValueError(f"lam must be nonnegative, got {lam}.")
    J = X0.shape[1]

    L = np.linalg.norm(X0, 2) ** 2  # Lipschitz constant of the gradient
    step = 1.0 / max(L, 1e-12)
    w = np.full(J, 1.0 / J)
    for _ in range(max_iter):
        grad = X0.T @ (X0 @ w - x1)
        w_new = np.maximum(w - step * (grad + lam), 0.0)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    support = np.flatnonzero(w > 1e-3 * w.sum())  # drop numerically-dead donors
    if support.size == 0:
        raise ValueError("lam too large: every donor weight shrank to zero.")
    ws = np.zeros(J)
    ws[support] = w[support] / w[support].sum()
    rmse = float(np.sqrt(np.mean((x1 - X0 @ ws) ** 2)))

    return RichResult(
        payload={
            "weights": ws,
            "support": support,
            "n_selected": int(support.size),
            "lam": lam,
            "rmse_pre": rmse,
            "method": "Synthetic control subset selection (nonnegative LASSO relax + simplex refit)",
        }
    )


def cheatsheet():
    return "causscss: nonneg-LASSO donor selection, simplex renormalise on support"
