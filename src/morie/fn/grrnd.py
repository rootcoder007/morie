# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Randomized hyperparameter search with K-fold cross-validated scoring."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_randomized_search_cv"]

_METHOD = "Randomized search with K-fold CV"


def _lcg_stream(seed):
    s = int(seed) % 2**32
    while True:
        s = (1664525 * s + 1013904223) % 2**32
        yield (s + 0.5) / 2**32


def geron_randomized_search_cv(X, y, param_dist, n_iter, K, fit_score=None, seed=42):
    r"""Sample configurations from distributions and score each by CV.

    Grid search spends its budget on the *product* of the grids, so
    adding one irrelevant hyperparameter multiplies the cost while
    teaching you nothing.  Random search spends a fixed budget of
    ``n_iter`` draws no matter how many dimensions there are, and it
    tries ``n_iter`` distinct values of *each* parameter rather than the
    handful a grid allows -- which is why it wins whenever only a few
    parameters actually matter.

    ``fit_score`` is required and its contract is enforced: given
    ``(X_train, y_train, X_val, y_val, params)`` it must return one
    finite scalar score, higher being better.  The folds are contiguous
    slices of the shuffled index, drawn from the reproducible LCG
    ``s = (1664525 s + 1013904223) mod 2^32``, so the same seed
    reproduces the search exactly.

    Parameters
    ----------
    X : array-like, shape (m, ...)
    y : array-like, shape (m,)
    param_dist : mapping
        ``name -> (low, high)`` for a uniform range, ``[a, b, c]`` for a
        discrete choice, or a callable ``f(u) -> value`` taking one
        uniform.
    n_iter : int
        Configurations to sample.
    K : int
        Folds, ``2 <= K <= m``.
    fit_score : callable
        ``fit_score(X_tr, y_tr, X_va, y_va, params) -> float``.
    seed : int, optional

    Returns
    -------
    RichResult
        Payload keys ``best_params``, ``best_score``, ``results``
        (params + mean/std score per configuration), ``fold_sizes``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 2, Randomized Search section.

    Examples
    --------
    A scorer that simply prefers small ``alpha`` -- the search should
    find the smallest sampled value:

    >>> X = [[0.0], [1.0], [2.0], [3.0]]
    >>> y = [0.0, 1.0, 2.0, 3.0]
    >>> f = lambda Xtr, ytr, Xva, yva, p: -p["alpha"]
    >>> r = geron_randomized_search_cv(X, y, {"alpha": (0.0, 10.0)}, n_iter=5, K=2,
    ...                                 fit_score=f, seed=1)
    >>> len(r["results"])
    5
    >>> round(r["best_score"], 6) == round(-r["best_params"]["alpha"], 6)
    True
    >>> r["best_score"] == max(row["mean_score"] for row in r["results"])
    True
    """
    A = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float).ravel()
    if A.ndim == 1:
        A = A[:, None]
    if A.size == 0:
        raise ValueError("X is empty.")
    if A.shape[0] != yv.size:
        raise ValueError(f"X has {A.shape[0]} rows but y has {yv.size} entries.")
    if fit_score is None or not callable(fit_score):
        raise ValueError(
            "fit_score is required and must be callable: "
            "fit_score(X_tr, y_tr, X_va, y_va, params) -> float."
        )
    n_iter = int(n_iter)
    K = int(K)
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}.")
    if not (2 <= K <= A.shape[0]):
        raise ValueError(f"K must lie in [2, {A.shape[0]}] (n_samples), got {K}.")
    if not isinstance(param_dist, dict) or not param_dist:
        raise ValueError("param_dist must be a non-empty mapping of name -> distribution.")

    rng = _lcg_stream(seed)
    perm = np.argsort([next(rng) for _ in range(A.shape[0])], kind="mergesort")
    folds = np.array_split(perm, K)
    if any(f.size == 0 for f in folds):
        raise ValueError(f"K={K} produces an empty fold for {A.shape[0]} samples.")

    def _sample(spec):
        u = next(rng)
        if callable(spec):
            return spec(u)
        if isinstance(spec, tuple) and len(spec) == 2 and all(
            isinstance(v, (int, float)) for v in spec
        ):
            lo, hi = float(spec[0]), float(spec[1])
            if not (hi > lo):
                raise ValueError(f"range ({lo}, {hi}) must have high > low.")
            return lo + u * (hi - lo)
        seq = list(spec)
        if not seq:
            raise ValueError("a discrete parameter list is empty.")
        return seq[min(int(u * len(seq)), len(seq) - 1)]

    results = []
    for _ in range(n_iter):
        params = {name: _sample(spec) for name, spec in param_dist.items()}
        scores = []
        for f in range(K):
            va = folds[f]
            tr = np.concatenate([folds[g] for g in range(K) if g != f])
            s = fit_score(A[tr], yv[tr], A[va], yv[va], params)
            s = float(s)
            if not np.isfinite(s):
                raise ValueError(f"fit_score returned a non-finite score for {params}.")
            scores.append(s)
        results.append(
            {
                "params": params,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "fold_scores": scores,
            }
        )

    best = max(results, key=lambda r: r["mean_score"])
    return RichResult(
        title="Randomized search CV",
        summary_lines=[("Configurations", n_iter), ("Folds", K),
                       ("Best score", best["mean_score"])],
        payload={
            "best_params": best["params"],
            "best_score": best["mean_score"],
            "results": results,
            "fold_sizes": [int(f.size) for f in folds],
            "estimate": best["mean_score"],
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grrnd: n_iter LCG draws from param_dist, K-fold CV per config; fit_score callable required"
