# morie.fn -- slice s04 (rootcoder007/morie)
"""Grid search for DNN hyperparameter tuning with cross-validation.

Book sections read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer.  Volume [Pages 427-476], Chapter 11, Section 11.4,
pp. 438-441: the tuning is a "full Cartesian grid search (with
sample = 1)" over the declared flags, run inside an inner cross-
validation, and the combination with the best inner score is kept.
Volume [Pages 109-139], Chapter 4, Section 4.4.2, p. 127, states the same
rule in general terms -- train "with every permutation of hyperparameter
choices using the training set.  Then, the combination of hyperparameters
with the best prediction performance on the validation set is chosen" --
and Section 4.3.2 with equation (4.1) supplies the score, CV_K, which is
the objective argmin_H CV_K(H) the function's docstring names.

DETERMINISM.  Folds are the in-order complementary partition, i mod K, so
K = n is exactly leave-one-out; the Cartesian product is enumerated in a
fixed order and ties go to the first point, so both arms select the same
combination.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["hyperparameter_tuning_grid"]


def _ridge_cv(X, y, K, params):
    n = len(y)
    lam = float(params.get("lam", 1.0))
    mse = []
    for f in range(K):
        tr = [i for i in range(n) if i % K != f]
        te = [i for i in range(n) if i % K == f]
        if not tr or not te:
            raise ValueError("hyperparameter_tuning_grid: a fold left no training or no testing rows")
        p = len(X[0])
        A = [[0.0] * p for _ in range(p)]
        b = [0.0] * p
        for i in tr:
            for a in range(p):
                b[a] += X[i][a] * y[i]
                for c in range(p):
                    A[a][c] += X[i][a] * X[i][c]
        for a in range(p):
            A[a][a] += lam
        beta = core.ridgesolve(A, b, 1e-12)
        s = 0.0
        for i in te:
            e = y[i]
            for a in range(p):
                e -= X[i][a] * beta[a]
            s += e * e
        mse.append(s / len(te))
    t = 0.0
    for v in mse:
        t += v
    return t / K


def _cartesian(grid):
    keys = list(grid.keys())
    out = [{}]
    for kname in keys:
        vals = list(grid[kname])
        if not vals:
            raise ValueError("hyperparameter_tuning_grid: %r has no candidate values" % (kname,))
        out = [dict(list(d.items()) + [(kname, v)]) for d in out for v in vals]
    return keys, out


def hyperparameter_tuning_grid(param_grid, cv_data, fit_cv=None, k=5):
    """Full Cartesian grid search scored by CV_K.

    Parameters
    ----------
    param_grid : mapping
        name -> candidate values.
    cv_data : pair
        (X, y); X is n-by-p, y has length n.
    fit_cv : callable, optional
        (X, y, K, params) -> CV score.  Defaults to a ridge regression
        whose penalty is the grid key "lam".
    k : int
        Number of inner folds; k = n is leave-one-out.

    Returns
    -------
    estimate    : the winning CV score
    best_params : the winning combination
    cv_score    : the same score
    scores      : the score of every grid point, in enumeration order
    grid        : the enumerated grid points
    """
    if not param_grid:
        raise ValueError("hyperparameter_tuning_grid: the grid is empty")
    X = core.mat(cv_data[0])
    y = core.vec(cv_data[1])
    n = len(y)
    if n < 2 or len(X) != n:
        raise ValueError("hyperparameter_tuning_grid: cv_data must be an n-by-p X and an n-vector y")
    K = int(k)
    if K < 2 or K > n:
        raise ValueError("hyperparameter_tuning_grid: k must lie between 2 and n")
    fn = fit_cv if fit_cv is not None else _ridge_cv
    keys, pts = _cartesian(param_grid)
    scores = [float(fn(X, y, K, pt)) for pt in pts]
    best = 0
    for i in range(1, len(scores)):
        if scores[i] < scores[best]:
            best = i
    return RichResult(
        title="Hyperparameter grid search",
        summary_lines=[("points", len(pts)), ("folds", K)],
        payload={
            "estimate": scores[best],
            "best_params": pts[best],
            "cv_score": scores[best],
            "scores": scores,
            "grid": pts,
            "keys": keys,
            "n": len(pts),
            "method": "argmin_H CV_K(H) over the full Cartesian grid, Chapter 11 Sect. 11.4 with CV_K of eq. (4.1)",
        },
    )


def cheatsheet():
    return "htprd: Grid search for DNN hyperparameter tuning with cross-validation"
