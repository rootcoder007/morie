# morie.fn -- function file (rootcoder007/morie)
"""LMS estimation error."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lms_estimation_error"]


def rangayyan_ch3_lms_estimation_error(x, w, r, n=None):
    r"""LMS instantaneous error (Rangayyan Ch. 3):

    .. math:: e(n) = x(n) - \mathbf{w}^T(n)\,\mathbf{r}(n),

    with r(n) the reference input. Note the weights carry a time
    index: in LMS they are updated at every sample, so this is the
    error under the CURRENT weights, not a fixed filter.

    Parameters
    ----------
    x : array-like, shape (N,)
        Primary input.
    w : array-like, shape (p,) or (N, p)
        Weights; a single vector is treated as time-invariant.
    r : array-like, shape (N, p)
        Reference input vectors.
    n : int, optional
        Index to report.

    Returns
    -------
    RichResult
        keys: ``error``, ``error_at_n``, ``mse``, ``time_varying``,
        ``N``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (the LMS algorithm).
    """
    x = np.asarray(x, dtype=float).ravel()
    R = np.atleast_2d(np.asarray(r, dtype=float))
    if R.shape[0] != x.size:
        R = R.T
    if R.shape[0] != x.size:
        raise ValueError("r must have one row per sample of x.")
    W = np.asarray(w, dtype=float)
    tv = W.ndim == 2
    if tv:
        if W.shape != R.shape:
            raise ValueError("time-varying w must match the shape of r.")
        est = np.sum(W * R, axis=1)
    else:
        W = np.atleast_1d(W).ravel()
        if W.size != R.shape[1]:
            raise ValueError(f"w has {W.size} weights but r has {R.shape[1]} columns.")
        est = R @ W
    e = x - est
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < e.size:
            raise ValueError(f"n must lie in 0..{e.size - 1}, got {idx}.")
        at_n = float(e[idx])
    return RichResult(payload={"error": e, "error_at_n": at_n,
                               "mse": float(np.mean(e**2)), "time_varying": bool(tv),
                               "N": int(x.size),
                               "method": "e(n) = x(n) - w^T(n) r(n); weights are time-indexed"})


def cheatsheet():
    return "rng156: LMS weights carry a time index; error is under CURRENT weights"
