# morie.fn -- function file (rootcoder007/morie)
"""General FIR filter."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_filter_general"]


def rangayyan_ch3_ma_filter_general(x, b_k, n=None, N=None):
    r"""General FIR (moving-average) filter (Rangayyan Ch. 3):

    .. math:: y(n) = \sum_{k=0}^{N} b_k\, x(n-k).

    The boxcar filters are the special case :math:`b_k = 1/M`. With
    arbitrary taps the response is no longer a sinc, and the filter is
    linear-phase only when the taps are symmetric -- which is checked
    and reported rather than assumed.

    Parameters
    ----------
    x : array-like
        Input.
    b_k : array-like
        Filter taps.
    n : int, optional
        Index to report.
    N : int, optional
        Interface compatibility (order taken from b_k).

    Returns
    -------
    RichResult
        keys: ``y``, ``y_at_n``, ``linear_phase`` (symmetric taps),
        ``dc_gain``, ``order``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    x = np.asarray(x, dtype=float).ravel()
    b = np.atleast_1d(np.asarray(b_k, dtype=float)).ravel()
    if b.size < 1:
        raise ValueError("b_k must be non-empty.")
    if x.size < b.size:
        raise ValueError(f"need at least {b.size} samples, got {x.size}.")
    y = np.convolve(x, b, mode="full")[: x.size]
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < y.size:
            raise ValueError(f"n must lie in 0..{y.size - 1}, got {idx}.")
        at_n = float(y[idx])
    return RichResult(payload={"y": y, "y_at_n": at_n,
                               "linear_phase": bool(np.allclose(b, b[::-1])),
                               "dc_gain": float(b.sum()),
                               "order": int(b.size - 1),
                               "method": "y(n) = sum b_k x(n-k); linear phase iff taps symmetric"})


def cheatsheet():
    return "rng087: general FIR; linear phase only for symmetric taps"
