# morie.fn -- k02 batch (rootcoder007/morie)
"""Median absolute deviation as a scale estimate.

Source consulted: Hampel, F.R. (1974), The influence curve and its role in
robust estimation, *JASA* 69(346), 383-393, section 5, where the MAD is
introduced as the scale estimate with the smallest possible gross-error
sensitivity and a breakdown point of 1/2.  The raw statistic is
``median(|x - median(x)|)``; multiplying by 1.4826 = 1/Phi^-1(3/4) makes it
consistent for sigma at the normal.  Matches ``stats::mad`` exactly.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["mad_scale"]


def mad_scale(x, constant=1.4826, center=None):
    """Median absolute deviation.

    Parameters
    ----------
    x : array-like
        Sample.
    constant : float, default 1.4826
        Consistency factor for the normal.
    center : float, optional
        Centre; the sample median if omitted.

    Returns
    -------
    RichResult
        estimate, raw_mad, center, constant, n, method.
    """
    v = np.asarray(x, dtype=float).ravel()
    ctr = float(np.median(v)) if center is None else float(center)
    raw = float(np.median(np.abs(v - ctr)))
    return RichResult(
        payload={
            "estimate": float(constant) * raw,
            "raw_mad": raw,
            "center": ctr,
            "constant": float(constant),
            "n": int(len(v)),
            "method": "Median absolute deviation scale estimate (Hampel 1974)",
        }
    )


# CANONICAL TEST
# >>> x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 9.9, 2.5, 3.3, 2.7]
# >>> r = mad_scale(x)
# >>> assert abs(r["estimate"] - 0.66717) < 1e-12     # stats::mad
# >>> assert abs(r["raw_mad"] - 0.45) < 1e-14


def cheatsheet():
    return "madsc(x): median absolute deviation scale estimate."


madscale = mad_scale
