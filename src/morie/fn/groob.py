# morie.fn -- function file (rootcoder007/morie)
"""Out-of-bag error for bagged ensembles."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_oob_evaluation"]


def geron_oob_evaluation(y, predictions, in_bag, task="regression"):
    r"""Out-of-bag error from per-estimator predictions and bag masks.

    Each bootstrap sample leaves out roughly

    .. math:: \lim_{m\to\infty}\left(1 - \tfrac1m\right)^{m} = e^{-1}
              \approx 0.368

    of the training rows, so every observation is out-of-bag for about
    a third of the estimators. Averaging only those gives a
    generalisation estimate with NO separate validation split -- the
    reason bagging can use all the data for fitting and still report an
    honest error.

    The estimate is not free of assumptions. It is nearly unbiased for
    a fixed ensemble size but tends to be pessimistic for the FULL
    ensemble, because each row is scored by only the ~37 % of trees
    that excluded it. ``mean_oob_votes`` shows how many that was;
    when it is small the estimate is noisy and biased upward.

    Parameters
    ----------
    y : array-like, shape (n,)
    predictions : array-like, shape (B, n)
        Prediction of estimator ``b`` for row ``i``.
    in_bag : array-like of bool, shape (B, n)
        Whether row ``i`` was in estimator ``b``'s bootstrap sample.
    task : {'regression', 'classification'}

    Returns
    -------
    RichResult
        ``oob_error``, ``oob_score``, ``oob_prediction``,
        ``coverage`` (fraction of rows with any OOB vote),
        ``mean_oob_votes``, ``expected_oob_fraction``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 7,
    out-of-bag evaluation. Breiman (1996), *Machine Learning*
    24:123-140.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0.0, 1.0])
    >>> # row 0 is out-of-bag for estimator 1, row 1 for estimator 0, so
    >>> # only the off-diagonal entries are ever scored
    >>> P = np.array([[9.0, 1.0], [0.0, 9.0]])
    >>> M = np.array([[True, False], [False, True]])
    >>> float(geron_oob_evaluation(y, P, M)["oob_error"])
    0.0
    """
    yv = np.asarray(y, dtype=float).ravel()
    P = np.atleast_2d(np.asarray(predictions, dtype=float))
    M = np.atleast_2d(np.asarray(in_bag)).astype(bool)
    n = yv.size
    if P.shape[1] != n or M.shape != P.shape:
        raise ValueError(
            "predictions and in_bag must both be (B, %d); got %s and %s."
            % (n, P.shape, M.shape)
        )
    if task not in ("regression", "classification"):
        raise ValueError(
            "task must be 'regression' or 'classification', got %r." % task
        )
    oob = ~M
    votes = oob.sum(axis=0)
    have = votes > 0
    pred = np.full(n, np.nan)
    if have.any():
        num = np.where(oob, P, 0.0).sum(axis=0)
        pred[have] = num[have] / votes[have]
    if task == "classification":
        pred_lab = np.where(np.isnan(pred), np.nan, (pred >= 0.5).astype(float))
        err = float(np.mean(pred_lab[have] != yv[have])) if have.any() else np.nan
        score = 1.0 - err if have.any() else np.nan
    else:
        err = float(np.mean((pred[have] - yv[have]) ** 2)) if have.any() \
            else np.nan
        var = float(np.var(yv[have])) if have.any() else np.nan
        score = 1.0 - err / var if var > 0 else np.nan
    return RichResult(
        payload={
            "estimate": err,
            "oob_error": err,
            "oob_score": score,
            "oob_prediction": pred,
            "coverage": float(np.mean(have)),
            "mean_oob_votes": float(np.mean(votes)),
            "expected_oob_fraction": float(np.exp(-1.0)),
            "task": task,
            "bias_note": (
                "each row is scored by only the estimators that excluded it, "
                "so the OOB error is pessimistic for the full ensemble; with "
                "few OOB votes per row it is also noisy"
            ),
            "n_estimators": int(P.shape[0]),
            "n": int(n),
            "method": "Out-of-bag evaluation",
        }
    )


def cheatsheet():
    return (
        "groob: out-of-bag error from bag masks, with the 1/e leave-out rate "
        "and the pessimism it carries"
    )
