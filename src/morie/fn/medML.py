# morie.fn -- function file (rootcoder007/morie)
"""Double machine-learning mediation (Neyman-orthogonal, cross-fitted)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ml_mediation_dml"]


def _ridge(X, y, lam=1e-3):
    D = np.column_stack([np.ones(X.shape[0]), X])
    A = D.T @ D + lam * np.eye(D.shape[1])
    A[0, 0] -= lam  # do not penalise the intercept
    return np.linalg.solve(A, D.T @ y)


def _pred(b, X):
    return np.column_stack([np.ones(X.shape[0]), X]) @ b


def ml_mediation_dml(x, m, y, c, n_folds=5, seed=0):
    r"""Cross-fitted partialling-out estimates of the mediation paths.

    Both paths are estimated after residualising on the covariates with
    sample splitting, the Robinson/Chernozhukov partialling-out step:

    .. math:: \tilde X = X - \hat E[X \mid C], \quad
              \tilde M = M - \hat E[M \mid C], \quad
              \tilde Y = Y - \hat E[Y \mid C],

    with each :math:`\hat E[\cdot \mid C]` fitted on the *other* folds.
    The a-path regresses :math:`\tilde M` on :math:`\tilde X`, the
    b- and direct paths regress :math:`\tilde Y` on
    :math:`(\tilde X, \tilde M)`. Cross-fitting removes the own-
    observation bias that makes naive plug-in machine learning
    estimates inconsistent.

    Parameters
    ----------
    x, m, y : array-like, shape (n,)
        Treatment, mediator, outcome.
    c : array-like, shape (n, p)
        Covariates (required -- with nothing to partial out use the
        plain product-of-coefficients estimator).
    n_folds : int, default 5
        Cross-fitting folds.
    seed : int, default 0
        Fold-assignment RNG seed.

    Returns
    -------
    RichResult
        keys: ``a``, ``b``, ``indirect``, ``direct``, ``total``,
        ``n_folds``, ``n``, ``method``.

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
    Hansen, C., Newey, W. & Robins, J. (2018). Double/debiased machine
    learning for treatment and structural parameters. *The
    Econometrics Journal*, 21(1), C1-C68.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    C = np.asarray(c, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    n = x.size
    if not (m.size == n and y.size == n and C.shape[0] == n):
        raise ValueError("x, m, y, c must share their first dimension.")
    k = int(n_folds)
    if not 2 <= k <= n // 2:
        raise ValueError(f"n_folds must lie in [2, {n // 2}], got {k}.")

    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % k
    rx, rm, ry = np.empty(n), np.empty(n), np.empty(n)
    for f in range(k):
        tr, te = folds != f, folds == f
        for src, dst in ((x, rx), (m, rm), (y, ry)):
            b = _ridge(C[tr], src[tr])
            dst[te] = src[te] - _pred(b, C[te])

    a = float(np.linalg.lstsq(rx[:, None], rm, rcond=None)[0][0])
    by, *_ = np.linalg.lstsq(np.column_stack([rx, rm]), ry, rcond=None)
    cprime, b = float(by[0]), float(by[1])

    return RichResult(
        payload={
            "a": a,
            "b": b,
            "indirect": a * b,
            "direct": cprime,
            "total": cprime + a * b,
            "n_folds": k,
            "n": int(n),
            "method": "Cross-fitted partialling-out mediation (DML)",
        }
    )


def cheatsheet():
    return "medML: residualise X, M, Y on C by cross-fitting, then a*b and c'"
