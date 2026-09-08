# morie.fn -- function file (rootcoder007/morie)
"""Upper record times and counts in a sequence."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['evtrecords', 'evt_record_count', 'evtrecordcount']


def evtrecords(x):
    """Upper record times and counts in a sequence.

    For any iid continuous sequence the record count depends on the distribution not at all -- only on the ranks -- so the mean and variance are exact functions of n alone and are returned next to the observed count. That is what makes records a distribution-free test for trend: a series with more records than about log n is not iid, whatever it is made of.


    Formula: R_n = sum_i 1{X_i > max(X_1..X_{i-1})}; E[R_n] = sum_i 1/i; Var[R_n] = sum_i (1/i - 1/i^2)

    Parameters
    ----------
    x : array-like
        The sequence, in observation order.

    Returns
    -------
    RichResult
        ``count``, ``times`` (indices of records), ``values``, ``expected``, ``variance``, ``z``, ``n``.

    References
    ----------
    Arnold, Balakrishnan and Nagaraja (1998), Records, Wiley.  Not held
    locally; the indicator representation and the resulting harmonic mean
    and variance are standard published results for iid continuous data.
    """
    x = C.vec(x)
    n = len(x)
    if n < 1:
        raise ValueError("need at least one observation")
    times, vals = [0], [x[0]]
    cur = x[0]
    for i in range(1, n):
        if x[i] > cur:
            cur = x[i]
            times.append(i); vals.append(x[i])
    ev = sum(1.0 / (i + 1) for i in range(n))
    vv = sum(1.0 / (i + 1) - 1.0 / ((i + 1) ** 2) for i in range(n))
    return RichResult(payload={
        "count": len(times), "times": times, "values": vals,
        "expected": ev, "variance": vv,
        "z": (len(times) - ev) / math.sqrt(vv) if vv > 0 else float("nan"),
        "n": n, "method": "Upper record times and counts"})


evt_record_count = evtrecords
evtrecordcount = evtrecords


def cheatsheet():
    return "evrec: Upper record times and counts in a sequence."
