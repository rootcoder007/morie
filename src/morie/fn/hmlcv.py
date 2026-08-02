# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Learning curves: RMSE on train and validation vs training set size."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_learning_curves"]

_METHOD = "Learning curves (train/validation RMSE vs training-set size)"


def _ols_fit(Xtr, ytr):
    theta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return theta


def _ols_predict(theta, Xte):
    return Xte @ theta


def geron_learning_curves(X, y, n_splits=10, val_fraction=0.2, fit=None, predict=None, seed=0):
    """
    Learning curves: RMSE on train and validation vs training set size.

    Formula: RMSE_train(m), RMSE_val(m) for increasing m

    Train on the first ``m`` rows, score on a fixed held-out set, and
    repeat for increasing ``m``.  The shape of the pair is the
    diagnostic:

    * both curves plateau *high* and close together -> underfitting;
      more data will not help, the model is too simple.
    * a wide, persistent gap with low training error -> overfitting;
      more data will help.

    The final gap is returned so this reading is a number rather than an
    eyeball.  Training error necessarily starts near zero -- a model can
    fit one or two points exactly -- and rises; validation error starts
    high and falls.  A training curve that does *not* rise usually means
    the split leaked.

    Parameters
    ----------
    X : array-like, shape (m, n)
        Design matrix (include a bias column yourself).
    y : array-like, shape (m,)
        Targets.
    n_splits : int
        Number of training-set sizes to evaluate.
    val_fraction : float
        Fraction held out for validation, in (0, 1).
    fit, predict : callable, optional
        ``fit(X, y) -> model`` and ``predict(model, X) -> y_pred``;
        default is OLS via ``lstsq``.
    seed : int
        Seed for the train/validation shuffle.

    Returns
    -------
    result : RichResult
        Keys: train_sizes, rmse_train, rmse_val, final_gap, verdict,
        estimate, n, method.

    Examples
    --------
    An exactly linear relation: once enough points are seen both errors
    are zero.

    >>> X = [[1.0, float(i)] for i in range(20)]
    >>> y = [3.0 + 2.0 * i for i in range(20)]
    >>> r = geron_learning_curves(X, y, n_splits=4, seed=0)
    >>> round(float(r["rmse_val"][-1]), 9)
    0.0
    >>> r["verdict"]
    'fits well'

    A model too simple for the data underfits: a constant-only design
    cannot follow a line, so both curves settle high and close.

    >>> Xc = [[1.0] for _ in range(20)]
    >>> u = geron_learning_curves(Xc, y, n_splits=4, seed=0)
    >>> bool(u["rmse_train"][-1] > 5) and u["verdict"] == 'underfitting'
    True

    Training sizes increase and the curves have one entry each:

    >>> len(r["train_sizes"]) == len(r["rmse_train"]) == len(r["rmse_val"])
    True
    >>> bool(np.all(np.diff(r["train_sizes"]) > 0))
    True

    References
    ----------
    Géron Ch 4
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yy = np.asarray(y, dtype=float).ravel()
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"geron_learning_curves: X must be a non-empty 2-D array, got shape {A.shape}")
    if A.shape[0] != yy.size:
        raise ValueError(f"geron_learning_curves: X has {A.shape[0]} rows but y has {yy.size} entries")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(yy)):
        raise ValueError("geron_learning_curves: X and y must be finite")
    vf = float(val_fraction)
    if not (0.0 < vf < 1.0):
        raise ValueError(f"geron_learning_curves: val_fraction must lie in (0, 1), got {val_fraction!r}")
    splits = int(n_splits)
    if splits < 2:
        raise ValueError(f"geron_learning_curves: n_splits must be at least 2, got {n_splits!r}")
    if (fit is None) != (predict is None):
        raise ValueError("geron_learning_curves: fit and predict must be supplied together")
    fit_fn = _ols_fit if fit is None else fit
    pred_fn = _ols_predict if predict is None else predict
    for name, fn in (("fit", fit_fn), ("predict", pred_fn)):
        if not callable(fn):
            raise ValueError(f"geron_learning_curves: {name} must be callable, got {type(fn).__name__}")

    m = A.shape[0]
    n_val = int(round(m * vf))
    if n_val < 1 or m - n_val < 2:
        raise ValueError(
            f"geron_learning_curves: {m} rows with val_fraction={vf} leaves {n_val} validation and "
            f"{m - n_val} training rows; both need to be usable"
        )
    perm = np.random.default_rng(int(seed)).permutation(m)
    val_i, tr_i = perm[:n_val], perm[n_val:]
    Xv, yv = A[val_i], yy[val_i]
    Xt, yt = A[tr_i], yy[tr_i]

    n_train = Xt.shape[0]
    sizes = np.unique(np.linspace(2, n_train, splits).astype(int))
    rmse_tr = []
    rmse_va = []
    for s in sizes:
        model = fit_fn(Xt[:s], yt[:s])
        p_tr = np.asarray(pred_fn(model, Xt[:s]), dtype=float).ravel()
        p_va = np.asarray(pred_fn(model, Xv), dtype=float).ravel()
        if p_tr.size != s or p_va.size != yv.size:
            raise ValueError(
                f"geron_learning_curves: predict returned {p_tr.size}/{p_va.size} values at training size {s}, "
                f"expected {s}/{yv.size}"
            )
        rmse_tr.append(float(np.sqrt(np.mean((p_tr - yt[:s]) ** 2))))
        rmse_va.append(float(np.sqrt(np.mean((p_va - yv) ** 2))))

    rmse_tr = np.asarray(rmse_tr)
    rmse_va = np.asarray(rmse_va)
    gap = float(rmse_va[-1] - rmse_tr[-1])
    scale = float(np.std(yy))
    if rmse_tr[-1] > 0.1 * max(scale, 1e-12) and gap < 0.5 * max(rmse_tr[-1], 1e-12):
        verdict = "underfitting"
    elif gap > max(0.5 * max(rmse_tr[-1], 1e-12), 0.1 * max(scale, 1e-12)):
        verdict = "overfitting"
    else:
        verdict = "fits well"

    return RichResult(
        title="Learning curves",
        summary_lines=[
            ("Training sizes", f"{int(sizes[0])} .. {int(sizes[-1])}"),
            ("Final train RMSE", float(rmse_tr[-1])),
            ("Final validation RMSE", float(rmse_va[-1])),
            ("Gap", gap),
            ("Verdict", verdict),
        ],
        interpretation=(
            "Curves that plateau high and together mean more data will not help; a wide persistent "
            "gap means it will."
        ),
        payload={
            "train_sizes": sizes,
            "rmse_train": rmse_tr,
            "rmse_val": rmse_va,
            "final_gap": gap,
            "verdict": verdict,
            "estimate": float(rmse_va[-1]),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlcv: learning curves -- train/validation RMSE vs training size, with an under/overfit verdict"
