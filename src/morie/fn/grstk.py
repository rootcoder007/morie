# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stacking meta-learner: base-model predictions feed a trained blender."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_stacking_predictor"]

_METHOD = "Stacking blender (least-squares meta-learner)"


def geron_stacking_predictor(base_preds, y, blender=None, include_intercept=True):
    r"""Fit the blender on out-of-fold base predictions.

    .. math::
        \hat y = g\bigl(h_1(x), \dots, h_L(x)\bigr)

    where :math:`g` is trained on predictions the base models made on
    data they did *not* see.  That "out of fold" qualifier is the whole
    method: fit the blender on in-sample base predictions and it learns
    to trust whichever model overfits hardest, because on training data
    that one looks perfect.  Nothing here can detect the mistake for you,
    so it is stated rather than assumed.

    The default blender is least squares over the base predictions (plus
    an intercept), which is Géron's linear blender; pass any callable
    ``blender(P, y) -> predict(P)`` to substitute your own, and the
    contract is enforced.

    Parameters
    ----------
    base_preds : array-like, shape (m, L)
        Column ``l`` holds model ``l``'s out-of-fold predictions.
    y : array-like, shape (m,)
    blender : callable, optional
        ``blender(P, y)`` returning a callable predictor.
    include_intercept : bool, optional

    Returns
    -------
    RichResult
        Payload keys ``predictions``, ``weights`` (blender
        coefficients, when linear), ``rmse``, ``base_rmse``,
        ``improvement``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 6, Stacking section.

    Examples
    --------
    Two base models, one biased high and one biased low; the blender
    recovers the truth that neither model gets alone:

    >>> P = [[1.5, 0.5], [2.5, 1.5], [3.5, 2.5]]
    >>> r = geron_stacking_predictor(P, [1.0, 2.0, 3.0])
    >>> [round(v, 6) for v in r["predictions"]]
    [1.0, 2.0, 3.0]
    >>> round(r["rmse"], 10)
    0.0

    Each base model on its own is worse:

    >>> [round(v, 6) for v in r["base_rmse"]]
    [0.5, 0.5]
    """
    P = np.atleast_2d(np.asarray(base_preds, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if P.ndim != 2 or P.size == 0:
        raise ValueError(f"base_preds must be a non-empty (m, L) matrix, got shape {P.shape}.")
    if P.shape[0] != yv.size:
        raise ValueError(f"base_preds has {P.shape[0]} rows but y has {yv.size} entries.")
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(yv)):
        raise ValueError("base_preds and y must be finite.")

    base_rmse = np.sqrt(np.mean((P - yv[:, None]) ** 2, axis=0))

    if blender is None:
        D = np.hstack([np.ones((P.shape[0], 1)), P]) if include_intercept else P
        coef, *_ = np.linalg.lstsq(D, yv, rcond=None)
        pred = D @ coef
        weights = coef.tolist()
    else:
        if not callable(blender):
            raise ValueError(f"blender must be callable, got {type(blender).__name__}.")
        fitted = blender(P, yv)
        if not callable(fitted):
            raise ValueError("blender(P, y) must return a callable predictor.")
        pred = np.asarray(fitted(P), dtype=float).ravel()
        if pred.shape != yv.shape:
            raise ValueError(
                f"blender predictor returned {pred.size} predictions for {yv.size} instances."
            )
        if not np.all(np.isfinite(pred)):
            raise ValueError("blender predictor returned non-finite values.")
        weights = None

    rmse = float(np.sqrt(np.mean((pred - yv) ** 2)))
    return RichResult(
        title="Stacking predictor",
        summary_lines=[("Base models", int(P.shape[1])), ("Blender RMSE", rmse)],
        payload={
            "predictions": pred.tolist(),
            "weights": weights,
            "rmse": rmse,
            "base_rmse": base_rmse.tolist(),
            "improvement": float(base_rmse.min() - rmse),
            "estimate": pred.tolist(),
            "n": int(P.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grstk: blender fit on OUT-OF-FOLD base predictions; default is least squares + intercept"
