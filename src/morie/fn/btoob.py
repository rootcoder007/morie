# morie.fn -- function file (rootcoder007/morie)
"""Out-of-bag error for a bootstrap-aggregated predictor."""

import numpy as np

from ._richresult import RichResult

__all__ = ["boot_oob_error"]


def boot_oob_error(x, y, fit_fn, predict_fn, B=100, loss=None, seed=0):
    r"""The out-of-bag error (Efron and Tibshirani 1997; Breiman
    1996b): fit the model on each bootstrap sample, and score each
    observation ONLY with the fits whose sample excluded it,

    .. math:: \widehat{\mathrm{Err}}_{oob} = \frac1n\sum_i
              \frac1{|C^{-i}|}\sum_{b \in C^{-i}}
              L\big(y_i, \hat f^{*b}(x_i)\big).

    Honesty is structural: no observation is ever scored by a fit
    that saw it, so the estimate needs no separate test set -- each
    point is out of bag for about 36.8% of the replicates, which is
    the (7.55) inclusion probability from the other side. The
    exclusion is asserted in the tests by comparing against the
    apparent error, which must be smaller. Observations in every bag
    (possible at small B) are dropped and counted, since a large
    count means B is too small rather than that the estimate is fine.

    ``fit_fn(x_boot, y_boot)`` returns a fitted object;
    ``predict_fn(fitted, x_new)`` returns predictions. Splitting the
    two lets the same machinery serve any model.

    Parameters
    ----------
    x : array-like, shape (n, p)
        Predictors.
    y : array-like, shape (n,)
        Response.
    fit_fn, predict_fn : callable
        The model, split into fit and predict.
    B : int, default 100
        Bootstrap replicates.
    loss : callable, optional
        Elementwise ``loss(y, yhat)``; squared error when omitted.
    seed : int, default 0
        Resampling seed.

    Returns
    -------
    RichResult
        keys: ``err_oob``, ``err_apparent``, ``per_observation``,
        ``n_dropped``, ``oob_fraction``, ``B``, ``n``, ``method``.

    References
    ----------
    Efron, B. and Tibshirani, R. (1997), *JASA* 92:548-560.
    Breiman, L. (1996), "Out-of-bag estimation", technical report,
    UC Berkeley.
    """
    from ._esl import squared_error

    A = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if A.shape[0] != yv.size:
        A = A.T
    if A.shape[0] != yv.size:
        raise ValueError(f"x has {A.shape[0]} rows for {yv.size} responses.")
    n = yv.size
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    Bn = int(B)
    if Bn < 1:
        raise ValueError(f"need at least one replicate, got {Bn}.")
    L = squared_error if loss is None else loss
    rng = np.random.default_rng(seed)
    loss_sum = np.zeros(n)
    oob_cnt = np.zeros(n)
    for _ in range(Bn):
        idx = rng.integers(0, n, n)
        fitted = fit_fn(A[idx], yv[idx])
        out = np.setdiff1d(np.arange(n), idx)
        if out.size:
            pred = np.asarray(predict_fn(fitted, A[out]), dtype=float).ravel()
            loss_sum[out] += np.asarray(L(yv[out], pred), dtype=float)
            oob_cnt[out] += 1
    keep = oob_cnt > 0
    per_i = np.where(keep, loss_sum / np.maximum(oob_cnt, 1), np.nan)
    err_oob = float(np.nanmean(per_i[keep])) if keep.any() else np.nan
    full = fit_fn(A, yv)
    err_app = float(np.mean(L(yv, np.asarray(
        predict_fn(full, A), dtype=float).ravel())))
    return RichResult(payload={
        "err_oob": err_oob, "err_apparent": err_app,
        "per_observation": per_i,
        "n_dropped": int((~keep).sum()),
        "oob_fraction": float(oob_cnt.mean() / Bn),
        "honesty_note": "no observation is ever scored by a fit that saw "
                        "it; each point is out of bag for about 36.8% of "
                        "replicates",
        "B": int(Bn), "n": int(n),
        "method": "Out-of-bag error (Efron-Tibshirani 1997; Breiman 1996)"})


def cheatsheet():
    return "btoob: score each point only with fits that never saw it -- honesty by construction"
