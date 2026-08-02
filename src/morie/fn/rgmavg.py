# morie.fn -- function file (rootcoder007/morie)
"""Moving-average filter."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_moving_average"]


def rangayyan_moving_average(x, M=8):
    r"""Causal moving-average (boxcar) filter (Rangayyan Ch. 3):

    .. math:: y[n] = \frac1M \sum_{k=0}^{M-1} x[n-k].

    A lowpass filter whose magnitude response is a sinc: it has zeros
    at multiples of fs/M, which is why an M chosen to place a zero on
    the interference frequency removes it exactly. The group delay is
    (M-1)/2 samples and is returned, because the output is NOT aligned
    with the input.

    Parameters
    ----------
    x : array-like
        Input signal.
    M : int, default 8
        Window length.

    Returns
    -------
    RichResult
        keys: ``y``, ``group_delay``, ``M``, ``N``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (moving-average filters).
    """
    x = np.asarray(x, dtype=float).ravel()
    M = int(M)
    if M < 1:
        raise ValueError(f"M must be at least 1, got {M}.")
    if x.size < M:
        raise ValueError(f"need at least M = {M} samples, got {x.size}.")
    y = np.convolve(x, np.ones(M) / M, mode="full")[: x.size]
    return RichResult(payload={"y": y, "group_delay": (M - 1) / 2.0, "M": M,
                               "N": int(x.size),
                               "method": "y[n] = (1/M) sum x[n-k]; sinc response, delay (M-1)/2"})


def cheatsheet():
    return "rgmavg: boxcar lowpass; zeros at fs/M multiples; delay (M-1)/2"
