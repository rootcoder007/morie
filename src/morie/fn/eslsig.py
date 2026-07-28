# morie.fn -- function file (rootcoder007/morie)
"""Residual variance for a linear model."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_residual_variance"]


def esl_residual_variance(X, y, beta=None):
    r"""Unbiased residual variance, ESL Eq. (3.8):

    .. math:: \hat\sigma^2 = \frac1{N - p - 1}\sum_{i=1}^N
              (y_i - \hat y_i)^2 .

    The book is explicit about why the denominator is
    :math:`N - p - 1` and not :math:`N`: it is what makes
    :math:`E(\hat\sigma^2) = \sigma^2`. The ``p + 1`` counts the
    ``p`` slopes plus the intercept, so a design matrix that already
    carries a column of ones must not be charged for it twice --
    ``intercept_in_X`` records which convention was detected.

    Dividing by ``N`` instead is the maximum-likelihood estimator and
    is biased downward by a factor :math:`(N-p-1)/N`; with ``p`` at
    all comparable to ``N`` that is not a rounding difference, so
    both are reported and the ratio is exact.

    Eq. (3.11) adds that :math:`(N-p-1)\hat\sigma^2 \sim
    \sigma^2\chi^2_{N-p-1}`, which is where ``df`` gets used.

    Parameters
    ----------
    x : array-like, shape (N, p) or (N, p+1)
        Design matrix, with or without a leading column of ones.
    y : array-like, shape (N,)
        Response.
    beta : array-like, optional
        Coefficients. Least squares is used when omitted.

    Returns
    -------
    RichResult
        keys: ``value`` (the (3.8) estimate), ``sigma``, ``rss``,
        ``df``, ``n``, ``p``, ``intercept_in_X``, ``mle_variance``,
        ``bias_factor``, ``fitted``, ``residuals``, ``method``.

    References
    ----------
    Hastie, Tibshirani and Friedman, *The Elements of Statistical
    Learning*, 2nd ed., Sec. 3.2, Eqs. (3.8) and (3.11). Read from
    the PDF.
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if A.shape[0] != yv.size:
        A = A.T
    if A.shape[0] != yv.size:
        raise ValueError(
            f"X has {A.shape[0]} rows for {yv.size} responses.")
    n = yv.size
    # an all-ones column IS the intercept; adding another would make
    # the design singular and charge a degree of freedom twice
    has_int = bool(np.any(np.all(np.isclose(A, 1.0), axis=0)))
    D = A if has_int else np.column_stack([np.ones(n), A])
    p = D.shape[1] - 1
    df = n - p - 1
    if df <= 0:
        raise ValueError(
            f"(3.8) needs N > p + 1; got N = {n} and p = {p}, so the "
            "residual degrees of freedom would be "
            f"{df} and the estimate is undefined.")
    if beta is None:
        b = np.linalg.lstsq(D, yv, rcond=None)[0]
    else:
        b = np.asarray(beta, dtype=float).ravel()
        if b.size == p:                      # slopes only
            b = np.r_[float(np.mean(yv - A @ b)), b] if not has_int else b
        if b.size != D.shape[1]:
            raise ValueError(
                f"beta has {b.size} entries for a design of "
                f"{D.shape[1]} columns.")
    fitted = D @ b
    resid = yv - fitted
    rss = float(resid @ resid)
    return RichResult(payload={
        "value": rss / df, "sigma": float(np.sqrt(rss / df)), "rss": rss,
        "df": int(df), "n": int(n), "p": int(p),
        "intercept_in_X": has_int,
        "mle_variance": rss / n,
        "bias_factor": float(df) / n,
        "fitted": fitted, "residuals": resid,
        "denominator_note": "N - p - 1, not N: that is what makes it unbiased (3.8)",
        "chi_square_fact": "(N-p-1) sigma_hat^2 ~ sigma^2 chi^2_{N-p-1} (3.11)",
        "method": "ESL (3.8) unbiased residual variance"})


def cheatsheet():
    return "eslsig: N - p - 1, not N -- the intercept costs a degree of freedom too"
