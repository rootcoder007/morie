# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grid search: evaluate every combination of hyperparameter values."""

import itertools

from . import _array_core as np

from ._richresult import RichResult
from .grcvs import geron_cross_validation_score

__all__ = ["geron_grid_search", "ridge_estimator"]

_METHOD = "Exhaustive grid search with K-fold cross-validation"


def ridge_estimator(X_train, y_train, alpha=0.0):
    """Default estimator: ridge regression in closed form.

    ``theta = (X^T X + alpha I)^-1 X^T y``.  Returns a ``predict``
    closure, which is the contract ``geron_grid_search`` expects of any
    estimator.  The bias column, if any, is penalised along with
    everything else -- pass a centred design if that matters.
    """
    X = np.atleast_2d(np.asarray(X_train, dtype=float))
    y = np.asarray(y_train, dtype=float).ravel()
    a = float(alpha)
    if a < 0:
        raise ValueError(f"ridge_estimator: alpha must be non-negative, got {alpha!r}")
    n = X.shape[1]
    theta = np.linalg.solve(X.T @ X + a * np.eye(n), X.T @ y)
    return lambda X_new: np.atleast_2d(np.asarray(X_new, dtype=float)) @ theta


def geron_grid_search(param_grid, X, y, estimator=None, K=3, score=None, shuffle=False, random_state=None):
    """
    Grid search: evaluate every combination of hyperparameter values.

    Formula: best = argmin over cartesian product of param values

    Every point of the Cartesian product is evaluated by K-fold
    cross-validation, delegated to
    :func:`morie.fn.grcvs.geron_cross_validation_score`.  Selection is by
    the *cross-validated* score, never the training score -- picking a
    hyperparameter on training error selects the least regularised
    option every time.

    The estimator contract is enforced: ``estimator(X_train, y_train,
    **params) -> predict``, where ``predict(X_test)`` returns one value
    per test row.  Anything else raises with the offending combination
    named.  The default estimator is :func:`ridge_estimator`, so a grid
    over ``{"alpha": [...]}`` works with no estimator supplied.

    The cost is the product of the grid sizes times ``K`` fits, and that
    number is returned -- grid search is exhaustive, which is its
    virtue and its expense.

    Parameters
    ----------
    param_grid : mapping of str to sequence
        Values to try for each hyperparameter.
    X : array-like, shape (m, n)
        Design matrix.
    y : array-like, shape (m,)
        Targets.
    estimator : callable, optional
        ``estimator(X_train, y_train, **params) -> predict``.
    K : int
        Folds per candidate.
    score : callable, optional
        ``score(y_true, y_pred) -> float``, higher is better; default R^2.
    shuffle : bool
        Shuffle before folding.
    random_state : int, optional
        Seed for the shuffle.

    Returns
    -------
    result : RichResult
        Keys: best_params, best_score, results, n_candidates, n_fits,
        estimate, n, method.

    Examples
    --------
    On noiseless data, the least regularised ridge wins:

    >>> X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
    >>> y = [3.0, 5.0, 7.0, 9.0]
    >>> r = geron_grid_search({"alpha": [0.0, 1.0, 100.0]}, X, y, K=2)
    >>> r["best_params"]
    {'alpha': 0.0}
    >>> round(r["best_score"], 8)
    1.0

    The grid is fully enumerated: 3 alphas x 2 folds = 6 fits.

    >>> r["n_candidates"], r["n_fits"]
    (3, 6)

    A two-parameter grid gives the Cartesian product:

    >>> est = lambda Xt, yt, a=0.0, b=0.0: (lambda Xn: np.asarray(Xn)[:, 1] * a + b)
    >>> g = geron_grid_search({"a": [1.0, 2.0], "b": [0.0, 1.0]}, X, y, estimator=est, K=2)
    >>> g["n_candidates"]
    4
    >>> g["best_params"]
    {'a': 2.0, 'b': 1.0}

    An estimator that returns the wrong number of predictions is named,
    not silently broadcast:

    >>> bad = lambda Xt, yt, a=0.0: (lambda Xn: np.zeros(1))
    >>> geron_grid_search({"a": [1.0]}, X, y, estimator=bad, K=2)
    Traceback (most recent call last):
        ...
    ValueError: geron_grid_search: candidate {'a': 1.0} failed: predict returned 1 predictions for fold 0, expected 2.

    References
    ----------
    Géron Ch 2
    """
    if not hasattr(param_grid, "items"):
        raise ValueError(f"geron_grid_search: param_grid must be a mapping name -> values, got {type(param_grid).__name__}")
    names = list(param_grid.keys())
    if not names:
        raise ValueError("geron_grid_search: param_grid is empty; there is nothing to search")
    value_lists = []
    for name in names:
        vals = list(param_grid[name])
        if not vals:
            raise ValueError(f"geron_grid_search: param_grid[{name!r}] is empty")
        value_lists.append(vals)

    est = ridge_estimator if estimator is None else estimator
    if not callable(est):
        raise ValueError(f"geron_grid_search: estimator must be callable, got {type(est).__name__}")

    A = np.atleast_2d(np.asarray(X, dtype=float))
    yy = np.asarray(y, dtype=float).ravel()
    if A.shape[0] != yy.size:
        raise ValueError(f"geron_grid_search: X has {A.shape[0]} rows but y has {yy.size} entries")

    results = []
    best = None
    for combo in itertools.product(*value_lists):
        params = dict(zip(names, combo))

        def _fit(Xtr, ytr, _p=params):
            pred = est(Xtr, ytr, **_p)
            if not callable(pred):
                raise ValueError(
                    f"estimator returned {type(pred).__name__}, expected a callable predict(X_test)"
                )
            return pred

        def _predict(model, Xte):
            return model(Xte)

        try:
            cv = geron_cross_validation_score(
                A, yy, K=K, fit=_fit, predict=_predict, score=score,
                shuffle=shuffle, random_state=random_state,
            )
        except ValueError as exc:
            raise ValueError(f"geron_grid_search: candidate {params} failed: {exc}") from None
        s = float(cv["cv_score"])
        results.append({"params": params, "cv_score": s, "fold_scores": cv["fold_scores"]})
        if best is None or s > best["cv_score"]:
            best = results[-1]

    n_cand = len(results)

    return RichResult(
        title="Grid search",
        summary_lines=[
            ("Candidates", n_cand),
            ("Folds", int(K)),
            ("Model fits", n_cand * int(K)),
            ("Best CV score", float(best["cv_score"])),
        ],
        tables=[
            {
                "title": "Candidates",
                "headers": ["params", "cv_score"],
                "rows": [[str(r["params"]), r["cv_score"]] for r in results],
            }
        ],
        interpretation=(
            "Selection is on the cross-validated score; selecting on the training score would "
            "always choose the least regularised candidate."
        ),
        payload={
            "best_params": best["params"],
            "best_score": float(best["cv_score"]),
            "results": results,
            "n_candidates": n_cand,
            "n_fits": n_cand * int(K),
            "estimate": float(best["cv_score"]),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmgrs: exhaustive grid search over the Cartesian product, scored by K-fold CV (delegates to grcvs)"
