# morie.fn -- function file (rootcoder007/morie)
"""11-point moving average."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_filter_11pt"]


from .rgmavg import rangayyan_moving_average


def rangayyan_ch3_ma_filter_11pt(x, n=None):
    r"""The 11-point moving-average filter (Rangayyan Ch. 3):

    .. math:: y(n) = \frac{1}{11} \sum_{k=0}^{10} x(n-k).

    The specific case used in the text for smoothing; delay is 5
    samples.

    Parameters
    ----------
    x : array-like
        Input.
    n : int, optional
        Index to report.

    Returns
    -------
    RichResult
        keys: ``y``, ``y_at_n``, ``group_delay`` (5.0), ``N``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    out = rangayyan_moving_average(x, M=11)
    y = out["y"]
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < y.size:
            raise ValueError(f"n must lie in 0..{y.size - 1}, got {idx}.")
        at_n = float(y[idx])
    return RichResult(payload={"y": y, "y_at_n": at_n, "group_delay": 5.0,
                               "N": int(y.size),
                               "method": "11-point moving average, delay 5 samples"})


def cheatsheet():
    return "rng039: M = 11 boxcar, delay 5"
