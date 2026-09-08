# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive grid search with K-fold cross-validation."""

import itertools

from . import _array_core as np

from ._richresult import RichResult
from .grkfd import geron_kfold_cv

__all__ = ["geron_grid_search_cv"]

_METHOD = "Exhaustive grid search with K-fold CV"


def geron_grid_search_cv(X, y, param_grid, K, fit_score, shuffle=False, seed=0):
    r"""Score every combination on every fold, keep the best mean.

    .. math::
        \text{best} = \arg\max_{\theta \in \text{Grid}}
        \frac{1}{K}\sum_k \mathrm{score}\bigl(\text{model}(\theta),
        \text{fold}_k\bigr)

    Cross-validated, not single-split: with one validation set the
    winner of a large grid is usually the combination that best fits
    that particular set's noise.  The per-combination standard deviation
    is reported for the same reason -- a configuration that wins by less
    than its own spread across folds has not really won.

    The grid is the Cartesian product of the lists in ``param_grid``, so
    the cost is ``K * prod(len(v))`` model fits; ``n_fits`` reports it.

    Splits come from :func:`morie.fn.grkfd.geron_kfold_cv`.  Model
    fitting is entirely the caller's: ``fit_score`` must be

    ``fit_score(X_train, y_train, X_val, y_val, params) -> float``

    returning a score where **higher is better** (negate an error).
    Its return value is checked for being a finite scalar.

    Parameters
    ----------
    X : array-like, shape (m, ...)
    y : array-like, shape (m,)
    param_grid : dict
        Name -> list of values. Must be non-empty with non-empty lists.
    K : int
        Folds, ``2 <= K <= m``.
    fit_score : callable
        See above.
    shuffle : bool, optional
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``best_params``, ``best_score``, ``best_index``,
        ``mean_scores``, ``std_scores``, ``all_scores`` (per fold),
        ``candidates``, ``n_fits``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 2, Grid Search section.

    Examples
    --------
    A scorer that peaks at ``a = 2`` (and ignores the data) picks it out
    of the grid, and the mean score is the value of the scorer:

    >>> def scorer(Xtr, ytr, Xva, yva, params):
    ...     return -abs(params["a"] - 2) - 0.1 * params["b"]
    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> r = geron_grid_search_cv(X, [0.0, 1.0, 2.0, 3.0],
    ...                          {"a": [1, 2, 3], "b": [0, 10]}, K=2,
    ...                          fit_score=scorer)
    >>> r["best_params"]
    {'a': 2, 'b': 0}
    >>> r["best_score"]
    0.0
    >>> r["n_fits"]
    12

    Every candidate is evaluated on every fold -- 6 combinations, 2
    folds:

    >>> len(r["candidates"]), len(r["all_scores"][0])
    (6, 2)
    """
    A = np.asarray(X)
    y_arr = np.asarray(y)
    m = A.shape[0]
    if y_arr.shape[0] != m:
        raise ValueError(f"y has {y_arr.shape[0]} rows but X has {m}.")
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("param_grid must be a non-empty dict of name -> list of values.")
    for name, vals in param_grid.items():
        if not isinstance(vals, (list, tuple, np.ndarray)) or len(vals) == 0:
            raise ValueError(f"param_grid[{name!r}] must be a non-empty sequence of values.")
    if not callable(fit_score):
        raise ValueError(f"fit_score must be callable, got {type(fit_score).__name__}.")

    names = list(param_grid)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*(param_grid[k] for k in names))]
    splits = geron_kfold_cv(m, K, shuffle=shuffle, seed=seed)["splits"]

    scores = np.empty((len(combos), len(splits)))
    for i, params in enumerate(combos):
        for k, (tr, va) in enumerate(splits):
            s = fit_score(A[tr], y_arr[tr], A[va], y_arr[va], params)
            s = np.asarray(s, dtype=float)
            if s.size != 1 or not np.isfinite(s):
                raise ValueError(
                    f"fit_score must return one finite number (higher is better); "
                    f"for params {params} on fold {k} it returned {s!r}."
                )
            scores[i, k] = float(s)

    mean = scores.mean(axis=1)
    std = scores.std(axis=1, ddof=1) if scores.shape[1] > 1 else np.zeros(len(combos))
    best = int(np.argmax(mean))

    return RichResult(
        title="Grid search (CV)",
        summary_lines=[("Candidates", len(combos)), ("Folds", len(splits)),
                       ("Best score", float(mean[best]))],
        payload={
            "best_params": combos[best],
            "best_score": float(mean[best]),
            "best_index": best,
            "best_std": float(std[best]),
            "mean_scores": mean.tolist(),
            "std_scores": std.tolist(),
            "all_scores": scores.tolist(),
            "candidates": combos,
            "n_fits": int(len(combos) * len(splits)),
            "estimate": combos[best],
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grgs: Cartesian grid x grkfd folds, caller supplies fit_score (higher better); reports spread"
