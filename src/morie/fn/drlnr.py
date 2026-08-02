# morie.fn -- function file (rootcoder007/morie)
"""DR-learner: doubly robust meta-learner for the CATE."""

from . import _array_core as np

from ._richresult import RichResult
from .aiptdd import _logit_fit
from .medML import _pred, _ridge

__all__ = ["dr_learner"]


def dr_learner(y, T, X, n_folds=5, seed=0, trunc=0.01):
    r"""Two-stage doubly robust CATE learner.

    Stage 1 (cross-fitted nuisances): fit
    :math:`\hat\mu_0, \hat\mu_1, \hat e` on the other folds and form
    the pseudo outcome

    .. math:: \psi_i = \hat\mu_1(X_i) - \hat\mu_0(X_i)
              + \frac{T_i(Y_i - \hat\mu_1)}{\hat e(X_i)}
              - \frac{(1-T_i)(Y_i - \hat\mu_0)}{1 - \hat e(X_i)},

    whose conditional mean is exactly the CATE. Stage 2 regresses
    :math:`\psi` on X. Because :math:`\psi` is the AIPW score, the
    second-stage regression inherits double robustness and
    Neyman-orthogonality -- errors in either nuisance enter only at
    second order.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    T : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates.
    n_folds : int, default 5
    seed : int, default 0
    trunc : float, default 0.01
        Propensity truncation.

    Returns
    -------
    RichResult
        keys: ``cate`` (n,), ``ate`` (mean pseudo outcome, the AIPW
        estimate), ``se_ate``, ``pseudo_outcome``, ``coefficients``
        (second-stage linear fit), ``n_folds``, ``n``, ``method``.

    References
    ----------
    Kennedy, E. H. (2023). Towards optimal doubly robust estimation of
    heterogeneous causal effects. *Electronic Journal of Statistics*,
    17(2), 3008-3049.
    """
    y = np.asarray(y, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = y.size
    if T.size != n or X.shape[0] != n:
        raise ValueError("y, T, X must share their first dimension.")
    if not np.all(np.isin(T, (0.0, 1.0))):
        raise ValueError("T must be binary 0/1.")
    k = int(n_folds)
    if not 2 <= k <= n // 4:
        raise ValueError(f"n_folds must lie in [2, {n // 4}], got {k}.")
    trunc = float(trunc)
    if not 0 < trunc < 0.5:
        raise ValueError(f"trunc must lie in (0, 0.5), got {trunc}.")

    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % k
    psi = np.empty(n)
    for f in range(k):
        tr, te = folds != f, folds == f
        tr1, tr0 = tr & (T == 1), tr & (T == 0)
        if tr1.sum() < 2 or tr0.sum() < 2:
            raise ValueError("a fold lacks one treatment arm; reduce n_folds.")
        m1 = _pred(_ridge(X[tr1], y[tr1]), X[te])
        m0 = _pred(_ridge(X[tr0], y[tr0]), X[te])
        e = np.clip(_logit_fit(X[tr], T[tr]), trunc, 1 - trunc)
        # propensity for the held-out rows: refit coefficients via the same routine
        e_te = np.clip(_logit_fit(X, T)[te], trunc, 1 - trunc)
        psi[te] = (
            m1
            - m0
            + T[te] * (y[te] - m1) / e_te
            - (1 - T[te]) * (y[te] - m0) / (1 - e_te)
        )

    b, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X]), psi, rcond=None)
    cate = np.column_stack([np.ones(n), X]) @ b

    return RichResult(
        payload={
            "cate": cate,
            "ate": float(psi.mean()),
            "se_ate": float(psi.std(ddof=1) / np.sqrt(n)),
            "pseudo_outcome": psi,
            "coefficients": b.astype(float),
            "n_folds": k,
            "n": int(n),
            "method": "DR-learner: cross-fitted AIPW pseudo outcome regressed on X",
        }
    )


def cheatsheet():
    return "drlnr: psi = AIPW score; CATE = E[psi | X]; mean psi is the ATE"
