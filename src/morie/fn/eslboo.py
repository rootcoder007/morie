# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap estimates of prediction error, ESL Sec. 7.11."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_bootstrap_err"]


def _least_squares(Xtr, ytr):
    """Default learner: ordinary least squares with an intercept."""
    D = np.column_stack([np.ones(len(ytr)), Xtr])
    b = np.linalg.lstsq(D, ytr, rcond=None)[0]
    return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ b


def esl_bootstrap_err(X, y, model=None, B=100, loss=None, seed=0):
    r"""The two bootstrap estimates of prediction error of ESL
    Sec. 7.11, reported together because the first one is misleading
    on its own.

    Eq. (7.54), the naive estimate:

    .. math:: \widehat{\mathrm{Err}}_{boot} = \frac1B\frac1N
              \sum_{b=1}^B \sum_{i=1}^N L(y_i, \hat f^{*b}(x_i)).

    **This is biased downward and the book says so.** The bootstrap
    datasets act as training samples and the original training set
    acts as the test sample, and the two overlap. The book's example
    is a 1-nearest-neighbour rule on a two-class problem with labels
    independent of the predictors: the true error rate is 0.5, but a
    contribution to (7.54) is only non-zero when observation ``i`` is
    absent from bootstrap sample ``b``, which by (7.55) happens with
    probability :math:`1 - 0.632 = 0.368`, so the expectation is
    :math:`0.5 \times 0.368 = 0.184` -- far below the truth.

    Eq. (7.56), the leave-one-out bootstrap, is the repair:

    .. math:: \widehat{\mathrm{Err}}^{(1)} = \frac1N\sum_{i=1}^N
              \frac1{|C^{-i}|}\sum_{b \in C^{-i}} L(y_i, \hat f^{*b}(x_i)),

    where :math:`C^{-i}` indexes the bootstrap samples NOT containing
    observation ``i``. It removes the overlap, at the price of a
    training-set-size bias in the other direction: the average number
    of distinct observations in a bootstrap sample is about
    :math:`0.632N`, so it behaves like twofold cross-validation and
    is biased UPWARD. Correcting that is what
    ``morie.fn.eslo63.esl_oob_632`` is for.

    Observations that appear in every bootstrap sample have an empty
    :math:`C^{-i}` and are dropped from (7.56), as the book allows;
    ``n_dropped`` reports how many, since a large count means ``B``
    is too small rather than that the estimate is fine.

    Parameters
    ----------
    x : array-like, shape (N, p)
        Predictors.
    y : array-like, shape (N,)
        Response.
    model : callable, optional
        ``model(X_train, y_train)`` returning a predict callable.
        Least squares when omitted.
    B : int, default 100
        Bootstrap replicates. The book uses 100.
    loss : callable, optional
        ``loss(y, yhat)`` elementwise; squared error when omitted.
    seed : int, default 0
        Resampling seed.

    Returns
    -------
    RichResult
        keys: ``err_boot`` (7.54), ``err_loo_boot`` (7.56),
        ``err_train``, ``inclusion_probability`` (7.55),
        ``optimistic`` (True), ``n_dropped``, ``B``, ``n``,
        ``which_to_use``, ``method``.

    References
    ----------
    Hastie, Tibshirani and Friedman, *The Elements of Statistical
    Learning*, 2nd ed., Sec. 7.11, Eqs. (7.54)-(7.56) and Fig. 7.12.
    Read from the PDF. Efron and Tibshirani (1997).
    """
    from ._esl import bootstrap_indices, inclusion_probability, squared_error

    A = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if A.shape[0] != yv.size:
        A = A.T
    if A.shape[0] != yv.size:
        raise ValueError(f"X has {A.shape[0]} rows for {yv.size} responses.")
    n = yv.size
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    fit = _least_squares if model is None else model
    L = squared_error if loss is None else loss
    idx = bootstrap_indices(n, B, seed=seed)
    Bn = idx.shape[0]

    preds = np.empty((Bn, n))
    for b in range(Bn):
        preds[b] = fit(A[idx[b]], yv[idx[b]])(A)
    losses = np.vstack([L(yv, preds[b]) for b in range(Bn)])

    err_boot = float(losses.mean())
    # C^{-i}: the replicates in which i does NOT appear
    inbag = np.zeros((Bn, n), dtype=bool)
    for b in range(Bn):
        inbag[b, idx[b]] = True
    oob = ~inbag
    counts = oob.sum(axis=0)
    keep = counts > 0
    per_i = np.where(keep, (losses * oob).sum(axis=0) / np.maximum(counts, 1),
                     np.nan)
    err_loo = float(np.nanmean(per_i[keep])) if keep.any() else np.nan
    err_train = float(np.mean(L(yv, fit(A, yv)(A))))
    return RichResult(payload={
        "err_boot": err_boot, "err_loo_boot": err_loo,
        "err_train": err_train,
        "inclusion_probability": inclusion_probability(n),
        "optimistic": True,
        "optimism_note": "(7.54) trains and tests on overlapping samples, so "
                         "it is biased DOWNWARD; use err_loo_boot (7.56)",
        "n_dropped": int((~keep).sum()),
        "per_observation": per_i,
        "which_to_use": "err_loo_boot for an honest estimate; feed it and "
                        "err_train to esl_oob_632 for the .632 correction",
        "B": int(Bn), "n": int(n),
        "method": "ESL (7.54) Err_boot and (7.56) leave-one-out bootstrap Err^(1)"})


def cheatsheet():
    return "eslboo: (7.54) overlaps train and test and is biased LOW -- (7.56) is the honest one"
