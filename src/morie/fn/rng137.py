# morie.fn -- function file (rootcoder007/morie)
"""Estimation error."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_estimation_error"]


def rangayyan_ch3_estimation_error(d, d_tilde, n=None):
    r"""Estimation error of an adaptive filter (Rangayyan Ch. 3):

    .. math:: e(n) = d(n) - \tilde d(n),

    the desired response minus its estimate. Also returns the mean
    squared error, the quantity the LMS and RLS recursions actually
    minimise.

    Parameters
    ----------
    d : array-like
        Desired response.
    d_tilde : array-like
        Estimate.
    n : int, optional
        Index to report; the whole series if omitted.

    Returns
    -------
    RichResult
        keys: ``error``, ``error_at_n``, ``mse``, ``N``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (adaptive filters).
    """
    d = np.asarray(d, dtype=float).ravel()
    dt = np.asarray(d_tilde, dtype=float).ravel()
    if d.size != dt.size:
        raise ValueError("d and d_tilde must have the same length.")
    if d.size < 1:
        raise ValueError("d must be non-empty.")
    e = d - dt
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < e.size:
            raise ValueError(f"n must lie in 0..{e.size - 1}, got {idx}.")
        at_n = float(e[idx])
    return RichResult(payload={"error": e, "error_at_n": at_n,
                               "mse": float(np.mean(e**2)), "N": int(e.size),
                               "method": "e(n) = d(n) - d_tilde(n)"})


def cheatsheet():
    return "rng137: e = d - d_tilde; MSE is what LMS/RLS minimise"
