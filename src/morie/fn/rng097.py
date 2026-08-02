# morie.fn -- function file (rootcoder007/morie)
"""8-point moving average."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_8point"]


from .rgmavg import rangayyan_moving_average


def rangayyan_ch3_ma_8point(x, n=None):
    r"""The 8-point moving-average filter (Rangayyan Ch. 3):

    .. math:: y(n) = \frac18 \sum_{k=0}^{7} x(n-k).

    Delay is 3.5 samples -- a non-integer, which is why an even-length
    boxcar cannot be delay-corrected by an integer shift.

    Parameters
    ----------
    x : array-like
        Input.
    n : int, optional
        Index to report.

    Returns
    -------
    RichResult
        keys: ``y``, ``y_at_n``, ``group_delay`` (3.5), ``N``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    out = rangayyan_moving_average(x, M=8)
    y = out["y"]
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < y.size:
            raise ValueError(f"n must lie in 0..{y.size - 1}, got {idx}.")
        at_n = float(y[idx])
    return RichResult(payload={"y": y, "y_at_n": at_n, "group_delay": 3.5,
                               "N": int(y.size),
                               "method": "8-point moving average, delay 3.5 (non-integer)"})


def cheatsheet():
    return "rng097: even M gives a half-sample delay, uncorrectable by integer shift"
