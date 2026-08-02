# morie.fn -- function file (rootcoder007/morie)
"""Bagging, ESL Sec. 8.7."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_bagging"]


def _least_squares(Xtr, ytr):
    D = np.column_stack([np.ones(len(ytr)), Xtr])
    b = np.linalg.lstsq(D, ytr, rcond=None)[0]
    return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ b


def wasserman_bagging(X, y, model=None, B=100, newdata=None, seed=0):
    r"""Bootstrap aggregation, ESL Sec. 8.7:

    .. math:: \hat f_{bag}(x) = \frac1B\sum_{b=1}^B \hat f^{*b}(x),

    the average of the fits over :math:`B` bootstrap samples.

    Bagging reduces VARIANCE and leaves bias where it was. Each
    :math:`\hat f^{*b}` is drawn from the same distribution, so the
    average has the same expectation as any one of them; only the
    spread changes. That has a sharp consequence the tests here check
    directly: **for a procedure that is linear in ``y`` -- least
    squares above all -- bagging does essentially nothing.** The
    bootstrap average of a linear fit converges to the fit on the
    original data as :math:`B` grows, so the whole exercise returns
    what it started with, minus some Monte-Carlo noise.

    Bagging pays off exactly where that argument fails: high-variance
    low-bias procedures whose output is a wildly nonlinear function
    of the data, of which a deep regression tree is the standard
    example. ``replicate_spread`` is the variance across the
    individual fits and ``bagged_spread`` is that divided by ``B``,
    the variance of the average -- which is the quantity the method
    is actually trying to move. ``max_shift_from_single_fit`` is the
    direct check on the linearity argument above: small for least
    squares, large for a deep tree.

    Out-of-bag predictions come free: each observation is out of bag
    for roughly 36.8% of the replicates, and averaging only those
    gives a fit that never saw the point it predicts.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Predictors.
    y : array-like, shape (n,)
        Response.
    model : callable, optional
        ``model(X_train, y_train)`` returning a predict callable.
        Least squares when omitted -- which is precisely the case
        bagging cannot help, and is the default only because it makes
        that visible.
    B : int, default 100
        Bootstrap replicates.
    newdata : array-like, optional
        Points to predict; the training rows when omitted.
    seed : int, default 0
        Resampling seed.

    Returns
    -------
    RichResult
        keys: ``prediction``, ``single_fit``, ``oob_prediction``,
        ``oob_mse``, ``replicate_spread``, ``bagged_spread``,
        ``n_oob_missing``, ``B``, ``n``, ``helps_when``, ``method``.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), *The Elements of
    Statistical Learning*, 2nd ed., Sec. 8.7. Read from the PDF.
    Breiman, L. (1996), "Bagging predictors", *Machine Learning*
    24:123-140.
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if A.shape[0] != yv.size:
        A = A.T
    if A.shape[0] != yv.size:
        raise ValueError(f"X has {A.shape[0]} rows for {yv.size} responses.")
    n = yv.size
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    Bn = int(B)
    if Bn < 1:
        raise ValueError(f"need at least one replicate, got {Bn}.")
    fit = _least_squares if model is None else model
    Q = A if newdata is None else np.atleast_2d(
        np.asarray(newdata, dtype=float))
    if Q.shape[1] != A.shape[1]:
        raise ValueError(
            f"newdata has {Q.shape[1]} columns, expected {A.shape[1]}.")

    rng = np.random.default_rng(seed)
    reps = np.empty((Bn, Q.shape[0]))
    oob_sum = np.zeros(n)
    oob_cnt = np.zeros(n)
    for b in range(Bn):
        idx = rng.integers(0, n, n)
        pred = fit(A[idx], yv[idx])
        reps[b] = pred(Q)
        out = np.setdiff1d(np.arange(n), idx)
        if out.size:
            oob_sum[out] += pred(A[out])
            oob_cnt[out] += 1
    bagged = reps.mean(axis=0)
    single = fit(A, yv)(Q)
    has_oob = oob_cnt > 0
    oob_pred = np.where(has_oob, oob_sum / np.maximum(oob_cnt, 1), np.nan)
    spread = float(np.mean(np.var(reps, axis=0)))
    return RichResult(payload={
        "prediction": bagged, "single_fit": single,
        "oob_prediction": oob_pred,
        "oob_mse": (float(np.mean((yv[has_oob] - oob_pred[has_oob]) ** 2))
                    if has_oob.any() and newdata is None else None),
        "replicate_spread": spread,
        # the Monte-Carlo variance of the average of B replicates,
        # which is what the averaging actually buys
        "bagged_spread": spread / Bn,
        "max_shift_from_single_fit": float(np.max(np.abs(bagged - single))),
        "n_oob_missing": int((~has_oob).sum()),
        "B": Bn, "n": int(n),
        "helps_when": "the base procedure is high-variance, low-bias and "
                      "NONLINEAR in y; for a linear procedure such as least "
                      "squares the bootstrap average converges back to the "
                      "original fit and bagging does essentially nothing",
        "leaves_bias_alone": "the replicates are identically distributed, so "
                             "the average has the same expectation as any one "
                             "of them; only the variance moves",
        "method": "Bagging, ESL Sec. 8.7: f_bag(x) = (1/B) sum_b f*b(x)"})


def cheatsheet():
    return "wsmbgn: bagging moves variance, never bias -- and does nothing at all for a linear fit"
