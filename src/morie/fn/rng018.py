# morie.fn -- function file (rootcoder007/morie)
"""Ensemble average function (Rangayyan eq. 3.18)."""


from math import fsum, sqrt

from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["ensavg", "rangayyan_ch3_ensemble_average_function"]


def ensavg(observations):
    """Ensemble average x_bar(t) over M records, at every instant.

    Rangayyan (2024) eq. (3.18):
        x_bar(t) = mu_x(t) = (1/M) sum_{k=1}^{M} x_k(t)   for all t.

    The book calls x_bar(t) a prototype of the random process and notes
    it is a filtered version of the M observations with diminished
    noise.  Records must be the same length -- an ensemble average across
    ragged records would silently average different numbers of traces at
    different instants.
    """
    recs = [aslist(r) for r in observations]
    m = len(recs)
    if m == 0:
        raise ValueError("need at least one observation")
    n = len(recs[0])
    if n == 0:
        raise ValueError("records must be nonempty")
    if any(len(r) != n for r in recs):
        raise ValueError("all records must have the same length")
    avg = [fsum(r[i] for r in recs) / m for i in range(n)]
    sd = [sqrt(fsum((r[i] - avg[i]) ** 2 for r in recs) / m)
          for i in range(n)]
    return RichResult(payload={
        "average": avg, "sd": sd, "m": m, "n": n,
        "se": [s / sqrt(m) for s in sd],
        "method": "Rangayyan (2024) eq. (3.18)"})


rangayyan_ch3_ensemble_average_function = ensavg  # pre-policy spelling


def cheatsheet():
    return "rng018: ensemble average function, Rangayyan eq. (3.18)"
