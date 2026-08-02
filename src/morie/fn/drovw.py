# morie.fn -- function file (rootcoder007/morie)
"""Doubly-robust ATE with overlap weighting."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["dr_overlap_weighted"]


def dr_overlap_weighted(y, D, X, ps=None, n_folds=2, seed=0):
    r"""Augmented IPW on overlap weights, with cross-fitting.

    .. math::
        \hat\tau = \frac{\sum_i h_i\left[\mu_1(x_i) - \mu_0(x_i)
            + \frac{d_i(y_i - \mu_1)}{e_i}
            - \frac{(1-d_i)(y_i - \mu_0)}{1-e_i}\right]}{\sum_i h_i},
        \qquad h_i = e_i(1-e_i).

    Doubly robust means consistent if **either** the outcome model or the
    propensity model is right, not both -- two chances rather than one. It is
    not a licence to be careless with either: when both are wrong the bias can
    exceed that of the simpler estimators it was built from.

    Overlap weighting is what keeps it stable. Plain AIPW inherits IPW's
    :math:`1/e` blow-up, so a single near-deterministic unit can swamp the
    estimate; :math:`h = e(1-e)` sends those weights to zero, at the cost of
    targeting the ATO rather than the ATE.

    Cross-fitting is not optional decoration. Fitting the nuisance models on
    the same data used for the final average induces overfitting bias that
    does not vanish with :math:`n`; splitting removes it.

    Parameters
    ----------
    y : array-like
        Outcome.
    D : array-like
        Treatment indicator, 0/1.
    X : array-like
        Covariates.
    ps : array-like, optional
        Known propensity scores. Estimated by logistic regression otherwise.
    n_folds : int
        Cross-fitting folds, at least 2.
    seed : int
        Seed for the fold split.

    Returns
    -------
    RichResult
        ``ate``, ``se``, ``ci``, ``estimand``, ``influence``,
        ``max_weight_share``.

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., et al. (2018).
        Double/debiased machine learning. *Econometrics Journal*, 21(1), C1-C68.

    Examples
    --------
    Recovers a known effect under confounding.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(3000, 2))
    >>> e = 1 / (1 + np.exp(-(0.8 * X[:, 0])))
    >>> D = (rng.random(3000) < e).astype(float)
    >>> y = 2.0 * D + X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, 3000)
    >>> r = dr_overlap_weighted(y, D, X)
    >>> bool(abs(r["ate"] - 2.0) < 0.2)
    True

    Overlap weighting keeps any single unit from dominating.

    >>> bool(r["max_weight_share"] < 0.02)
    True

    The estimand is the ATO, and that is stated rather than left implied.

    >>> str(r["estimand"])
    'ATO (overlap-weighted)'
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    D = np.atleast_1d(np.asarray(D, dtype=float)).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = y.size
    if not (D.size == n and X.shape[0] == n):
        raise ValueError("y, D and X must agree on the number of observations")
    if not np.all((D == 0) | (D == 1)):
        raise ValueError("D must be 0/1")
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    rng = np.random.default_rng(seed)
    fold = np.empty(n, dtype=int)
    idx = rng.permutation(n)
    fold[idx] = np.arange(n) % n_folds

    A = np.column_stack([np.ones(n), X])
    e_hat = np.empty(n)
    mu1 = np.empty(n)
    mu0 = np.empty(n)
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        if ps is None:
            b = np.zeros(A.shape[1])
            for _ in range(200):
                p = 1 / (1 + np.exp(-np.clip(A[tr] @ b, -500, 500)))
                b -= 0.5 * (A[tr].T @ (p - D[tr]) / max(tr.sum(), 1) + 1e-4 * b)
            e_hat[te] = 1 / (1 + np.exp(-np.clip(A[te] @ b, -500, 500)))
        else:
            e_hat[te] = np.asarray(ps, dtype=float).ravel()[te]
        for lvl, out in ((1, mu1), (0, mu0)):
            m = tr & (D == lvl)
            if m.sum() > A.shape[1]:
                c = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
                out[te] = A[te] @ c
            else:
                out[te] = y[m].mean() if m.any() else y[tr].mean()
    e_hat = np.clip(e_hat, 1e-4, 1 - 1e-4)
    h = e_hat * (1.0 - e_hat)
    psi = (mu1 - mu0
           + D * (y - mu1) / e_hat
           - (1 - D) * (y - mu0) / (1 - e_hat))
    tot = float(h.sum())
    ate = float(np.sum(h * psi) / tot)
    infl = h * (psi - ate) / (tot / n)
    se = float(np.std(infl, ddof=1) / np.sqrt(n))
    return RichResult(
        title="Doubly-robust ATE (overlap-weighted)",
        summary_lines=[("n", n), ("ATE", ate), ("se", se)],
        warnings=["doubly robust means consistent if EITHER nuisance model is "
                  "right, not that both may be wrong"],
        payload={
            "ate": ate, "se": se,
            "ci": (ate - 1.96 * se, ate + 1.96 * se),
            "estimand": "ATO (overlap-weighted)", "influence": infl,
            "propensity": e_hat, "mu1": mu1, "mu0": mu0,
            "max_weight_share": float(h.max() / tot),
            "n_folds": n_folds, "method": "dr_overlap_weighted",
        },
    )


def cheatsheet():
    return "drovw: DR = either model right, not both wrong; overlap weights stop 1/e blow-up; cross-fitting is required"
