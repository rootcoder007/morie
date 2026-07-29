# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hyperparameter tuning: optimize over discrete grid or random samples."""

import numpy as np

from ._richresult import RichResult
from .grcvs import geron_cross_validation_score
from .hmgrs import geron_grid_search, ridge_estimator

__all__ = ["geron_hyperparameter_tuning"]

_METHOD = "Hyperparameter tuning by grid or random search"


def geron_hyperparameter_tuning(param_grid, X, y, estimator=None, search="grid", n_iter=10, K=3, seed=0, score=None):
    """
    Hyperparameter tuning: optimize over discrete grid or random samples.

    Formula: theta* = argmin_{theta in H} CV(theta)

    Exhaustive grid search is delegated to
    :func:`morie.fn.hmgrs.geron_grid_search`; what this entry adds is the
    random-search alternative and the comparison between them.

    Random search wins when only a few hyperparameters matter, which is
    usually.  A grid of ``g`` values per parameter over ``p`` parameters
    tries only ``g`` distinct values of the one parameter that matters,
    at a cost of ``g^p`` fits; ``n_iter`` random draws try ``n_iter``
    distinct values of it for ``n_iter`` fits.  Both counts are
    returned, along with ``distinct_values_per_param``, so the
    comparison is concrete.

    The estimator contract is the same as for grid search:
    ``estimator(X_train, y_train, **params) -> predict``.  The default is
    ridge regression, so ``{"alpha": [...]}`` needs no estimator.

    Parameters
    ----------
    param_grid : mapping of str to sequence
        Candidate values per hyperparameter.
    X : array-like, shape (m, n)
        Design matrix.
    y : array-like, shape (m,)
        Targets.
    estimator : callable, optional
        As in :func:`geron_grid_search`.
    search : {"grid", "random"}
        Exhaustive or sampled.
    n_iter : int
        Number of draws when ``search="random"``.
    K : int
        Folds per candidate.
    seed : int
        Seed for the random draws.
    score : callable, optional
        ``score(y_true, y_pred) -> float``, higher is better.

    Returns
    -------
    result : RichResult
        Keys: best_params, best_score, results, n_candidates, n_fits,
        distinct_values_per_param, estimate, n, method.

    Examples
    --------
    Grid search on noiseless data picks the least regularised ridge:

    >>> X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
    >>> y = [3.0, 5.0, 7.0, 9.0]
    >>> r = geron_hyperparameter_tuning({"alpha": [0.0, 1.0, 100.0]}, X, y, K=2)
    >>> r["best_params"], round(r["best_score"], 8)
    ({'alpha': 0.0}, 1.0)

    Random search over the same grid costs what you ask it to, and each
    draw is an independent combination:

    >>> s = geron_hyperparameter_tuning({"alpha": [0.0, 1.0, 100.0]}, X, y,
    ...                                 search="random", n_iter=4, K=2, seed=0)
    >>> s["n_candidates"], s["n_fits"]
    (4, 8)

    The cost argument in numbers: a 5-value grid over 3 parameters is
    125 candidates but still only 5 distinct values of each; 20 random
    draws are 20 candidates and up to 5 distinct values each here,
    because the pool itself has only 5.

    >>> big = {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [0.0, 1.0, 2.0, 3.0, 4.0],
    ...        "c": [0.0, 1.0, 2.0, 3.0, 4.0]}
    >>> est = lambda Xt, yt, a=1.0, b=0.0, c=0.0: (lambda Xn: np.asarray(Xn)[:, 1] * a + b + c)
    >>> gs = geron_hyperparameter_tuning(big, X, y, estimator=est, K=2)
    >>> gs["n_candidates"]
    125
    >>> rs = geron_hyperparameter_tuning(big, X, y, estimator=est, search="random",
    ...                                  n_iter=20, K=2, seed=1)
    >>> rs["n_candidates"]
    20

    References
    ----------
    Géron Ch 1
    """
    if search not in ("grid", "random"):
        raise ValueError(f"geron_hyperparameter_tuning: search must be 'grid' or 'random', got {search!r}")
    if not hasattr(param_grid, "items"):
        raise ValueError(
            f"geron_hyperparameter_tuning: param_grid must be a mapping name -> values, "
            f"got {type(param_grid).__name__}"
        )
    names = list(param_grid.keys())
    if not names:
        raise ValueError("geron_hyperparameter_tuning: param_grid is empty")
    pools = {}
    for name in names:
        vals = list(param_grid[name])
        if not vals:
            raise ValueError(f"geron_hyperparameter_tuning: param_grid[{name!r}] is empty")
        pools[name] = vals
    distinct = {name: len({repr(v) for v in vals}) for name, vals in pools.items()}

    if search == "grid":
        inner = geron_grid_search(param_grid, X, y, estimator=estimator, K=K, score=score)
        payload = dict(inner.payload)
        payload["distinct_values_per_param"] = distinct
        payload["search"] = "grid"
        payload["method"] = _METHOD
        return RichResult(
            title="Hyperparameter tuning (grid)",
            summary_lines=inner.summary_lines,
            tables=inner.tables,
            interpretation=(
                "Exhaustive: the cost is the product of the pool sizes, but only "
                f"{max(distinct.values())} distinct values of any single parameter get tried."
            ),
            payload=payload,
        )

    iters = int(n_iter)
    if iters < 1:
        raise ValueError(f"geron_hyperparameter_tuning: n_iter must be at least 1, got {n_iter!r}")
    est = ridge_estimator if estimator is None else estimator
    if not callable(est):
        raise ValueError(f"geron_hyperparameter_tuning: estimator must be callable, got {type(est).__name__}")

    A = np.atleast_2d(np.asarray(X, dtype=float))
    yy = np.asarray(y, dtype=float).ravel()
    if A.shape[0] != yy.size:
        raise ValueError(f"geron_hyperparameter_tuning: X has {A.shape[0]} rows but y has {yy.size} entries")

    rng = np.random.default_rng(int(seed))
    results = []
    best = None
    for _ in range(iters):
        params = {name: pools[name][int(rng.integers(len(pools[name])))] for name in names}

        def _fit(Xtr, ytr, _p=params):
            pred = est(Xtr, ytr, **_p)
            if not callable(pred):
                raise ValueError(f"estimator returned {type(pred).__name__}, expected a callable predict(X_test)")
            return pred

        try:
            cv = geron_cross_validation_score(A, yy, K=K, fit=_fit, predict=lambda mdl, Xte: mdl(Xte), score=score)
        except ValueError as exc:
            raise ValueError(f"geron_hyperparameter_tuning: candidate {params} failed: {exc}") from None
        s = float(cv["cv_score"])
        results.append({"params": params, "cv_score": s, "fold_scores": cv["fold_scores"]})
        if best is None or s > best["cv_score"]:
            best = results[-1]

    return RichResult(
        title="Hyperparameter tuning (random)",
        summary_lines=[
            ("Draws", iters),
            ("Folds", int(K)),
            ("Model fits", iters * int(K)),
            ("Best CV score", float(best["cv_score"])),
        ],
        interpretation=(
            "Random search spends its budget on distinct values of each parameter rather than on "
            "the Cartesian product, which is why it beats a grid when only a few parameters matter."
        ),
        payload={
            "best_params": best["params"],
            "best_score": float(best["cv_score"]),
            "results": results,
            "n_candidates": iters,
            "n_fits": iters * int(K),
            "distinct_values_per_param": distinct,
            "search": "random",
            "estimate": float(best["cv_score"]),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmhpt: hyperparameter tuning -- grid (delegates to hmgrs) or random search, both CV-scored"
