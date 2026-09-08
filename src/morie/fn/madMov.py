# morie.fn -- k02 batch (rootcoder007/morie)
"""Moving (rolling) median absolute deviation.

Source consulted: Hampel, F.R. (1974), *JASA* 69(346), 383-393.  The MAD is
applied to each trailing window of length ``window``, giving a local scale
track whose breakdown point inside the window is still 1/2 -- the property
that makes it usable on a series that contains bursts.  Output element j
covers ``x[j:j+window]``, so there are ``n - window + 1`` of them.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["moving_mad"]


def moving_mad(x, window, constant=1.4826):
    """Rolling median absolute deviation.

    Parameters
    ----------
    x : array-like
        Series.
    window : int
        Window length.
    constant : float, default 1.4826
        Consistency factor for the normal.

    Returns
    -------
    RichResult
        estimate (last window), values, centers, window, n, method.
    """
    v = np.asarray(x, dtype=float).ravel()
    w = int(window)
    n = len(v)
    vals = []
    ctrs = []
    for j in range(0, n - w + 1):
        seg = v[j : j + w]
        c = float(np.median(seg))
        vals.append(float(constant) * float(np.median(np.abs(seg - c))))
        ctrs.append(c)
    return RichResult(
        payload={
            "estimate": vals[-1] if vals else float("nan"),
            "values": vals,
            "centers": ctrs,
            "window": w,
            "n": int(n),
            "method": "Moving median absolute deviation (Hampel 1974)",
        }
    )


# CANONICAL TEST
# >>> x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# >>> r = moving_mad(x, 3)
# >>> assert len(r["values"]) == 4
# >>> assert abs(r["values"][0] - 1.4826) < 1e-12   # MAD of 1,2,3 is 1


def cheatsheet():
    return "madMov(x, window): rolling median absolute deviation."


movingmad = moving_mad
