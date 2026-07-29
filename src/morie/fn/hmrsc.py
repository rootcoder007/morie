# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Randomized hyperparameter search."""

import numpy as np

from ._richresult import RichResult
from .grcvs import geron_cross_validation_score

__all__ = ["geron_randomized_search"]


def _ridge_estimator(params):
    """Default estimator: ridge regression with an ``alpha`` hyperparameter."""
    alpha = float(params.get("alpha", 0.0))
    if alpha < 0:
        raise ValueError(f"geron_randomized_search: alpha must be non-negative, got {alpha}")

    def fit(Xtr, ytr, _a=alpha):
        G = Xtr.T @ Xtr + _a * np.eye(Xtr.shape[1])
        return np.linalg.solve(G, Xtr.T @ ytr)

    def predict(theta, Xte):
        return Xte @ theta

    return fit, predict


def geron_randomized_search(param_dist, n_iter, X, y, estimator=None, K=3, seed=0, score=None):
    """
    Randomized search: sample n_iter hyperparameter combinations from distributions.

    Formula: theta_i ~ prior; pick best over n_iter samples

    Grid search spends its budget on the CROSS PRODUCT of the grids, so
    adding an irrelevant hyperparameter multiplies the cost while the
    important one is still explored at its handful of grid values.
    Random search spends the same budget on n_iter distinct values of
    EVERY parameter, which is why it wins as soon as the parameters
    differ in importance -- and the budget is set directly, not implied
    by the grid.

    Each candidate is scored by K-fold cross-validation DELEGATED to
    :func:`morie.fn.grcvs.geron_cross_validation_score`.

    ``param_dist`` maps a name to a list (uniform choice), a ``(lo, hi)``
    tuple (uniform on the interval) or a callable ``f(u)`` receiving a
    uniform u in (0, 1).

    Parameters
    ----------
    param_dist : mapping
        Search space as described.
    n_iter : int
        Candidates to draw (>= 1).
    X : array-like, shape (m, n)
    y : array-like, shape (m,)
    estimator : callable, optional
        ``estimator(params) -> (fit, predict)`` matching the
        cross-validation contract. Defaults to ridge regression on
        ``alpha``.
    K : int, default 3
        Folds.
    seed : int, default 0
        Integer-LCG seed for the draws.
    score : callable, optional
        ``score(y_true, y_pred) -> float``, higher better. Default R^2.

    Returns
    -------
    result : RichResult
        Keys: best_params, best_score, candidates, scores, estimate, n,
        method.

    Examples
    --------
    On noiseless y = 2x, no shrinkage is best and it fits perfectly:

    >>> X = [[1.0], [2.0], [3.0], [4.0]]
    >>> r = geron_randomized_search({"alpha": [0.0, 100.0]}, 6, X, [2.0, 4.0, 6.0, 8.0], K=2)
    >>> float(r["best_params"]["alpha"]), round(float(r["best_score"]), 8)
    (0.0, 1.0)
    >>> len(r["candidates"]), len(r["scores"])
    (6, 6)

    Interval and callable specifications are honoured:

    >>> r2 = geron_randomized_search({"alpha": (0.0, 1.0)}, 4, X, [2.0, 4.0, 6.0, 8.0], K=2)
    >>> bool(all(0.0 <= c["alpha"] <= 1.0 for c in r2["candidates"]))
    True

    References
    ----------
    Geron Ch 2
    """
    if not hasattr(param_dist, "items"):
        raise ValueError("geron_randomized_search: param_dist must be a mapping of name -> distribution")
    space = dict(param_dist)
    if not space:
        raise ValueError("geron_randomized_search: param_dist is empty")
    N = int(n_iter)
    if N < 1:
        raise ValueError(f"geron_randomized_search: n_iter must be >= 1, got {n_iter!r}")
    est = _ridge_estimator if estimator is None else estimator
    if not callable(est):
        raise ValueError("geron_randomized_search: estimator must be callable")

    s = int(seed) % 2**32

    def _u():
        nonlocal s
        s = (1664525 * s + 1013904223) % 2**32
        return (s + 0.5) / 2**32

    candidates = []
    for _ in range(N):
        params = {}
        for name, spec in space.items():
            u = _u()
            if callable(spec):
                params[name] = spec(u)
            elif isinstance(spec, tuple) and len(spec) == 2 and all(np.ndim(v) == 0 for v in spec):
                lo, hi = float(spec[0]), float(spec[1])
                if not (hi > lo):
                    raise ValueError(f"geron_randomized_search: interval for {name!r} must have hi > lo, got {spec!r}")
                params[name] = lo + u * (hi - lo)
            elif isinstance(spec, (list, np.ndarray)):
                opts = list(spec)
                if not opts:
                    raise ValueError(f"geron_randomized_search: the choice list for {name!r} is empty")
                params[name] = opts[min(int(u * len(opts)), len(opts) - 1)]
            else:
                raise ValueError(
                    f"geron_randomized_search: distribution for {name!r} must be a list, a (lo, hi) tuple "
                    f"or a callable, got {type(spec).__name__}"
                )
        candidates.append(params)

    scores = []
    for i, params in enumerate(candidates):
        built = est(params)
        try:
            fit, predict = built
        except (TypeError, ValueError):
            raise ValueError(f"geron_randomized_search: estimator must return a (fit, predict) pair, got {built!r}") from None
        cv = geron_cross_validation_score(X, y, K=int(K), fit=fit, predict=predict, score=score)
        scores.append(float(cv["cv_score"]))
        del i

    best = int(np.argmax(scores))
    return RichResult(
        title="Randomized hyperparameter search",
        summary_lines=[("Candidates", N), ("Folds", int(K)), ("Best CV score", scores[best])],
        interpretation="The budget buys n_iter values of every parameter, not the cross product of grids.",
        payload={
            "best_params": candidates[best],
            "best_score": scores[best],
            "best_index": best,
            "candidates": candidates,
            "scores": scores,
            "estimate": scores[best],
            "n": int(np.asarray(y).size),
            "method": "Randomized search scored by K-fold CV (delegated to morie.fn.grcvs)",
        },
    )


def cheatsheet():
    return "hmrsc: Randomized hyperparameter search over sampled candidates"
