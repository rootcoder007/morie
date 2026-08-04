# morie.fn -- function file (rootcoder007/morie)
"""Ensemble mean at one instant (Rangayyan eq. 3.15)."""


from math import fsum, sqrt

from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["ensmean", "rangayyan_ch3_ensemble_mean"]


def ensmean(observations, index=None):
    """Ensemble mean of M observations at a single instant t1.

    Rangayyan (2024) eq. (3.15):
        mu_x(t1) = lim_{M->inf} (1/M) sum_{k=1}^{M} x_k(t1).

    Parameters
    ----------
    observations : sequence
        Either the M values already sampled at t1, or M whole records
        from which ``index`` selects the instant.
    index : int, optional
        Sample index t1 within each record.

    Notes
    -----
    The SE of the ensemble mean is sigma/sqrt(M): the 1/sqrt(M) noise
    reduction the book attributes to synchronized averaging in Section
    3.3.1 is exactly this, read one instant at a time.
    """
    if index is None:
        vals = aslist(observations)
    else:
        i = int(index)
        vals = []
        for rec in observations:
            r = aslist(rec)
            if i < 0 or i >= len(r):
                raise IndexError("index %d outside a record of length %d"
                                 % (i, len(r)))
            vals.append(r[i])
    m = len(vals)
    if m == 0:
        raise ValueError("need at least one observation")
    mu = fsum(vals) / m
    var = fsum((v - mu) ** 2 for v in vals) / m
    return RichResult(payload={
        "mean": mu, "m": m, "sd": sqrt(var),
        "se": sqrt(var / m) if m > 0 else float("nan"),
        "method": "Rangayyan (2024) eq. (3.15)"})


rangayyan_ch3_ensemble_mean = ensmean  # pre-policy spelling


def cheatsheet():
    return "rng015: ensemble mean at an instant, Rangayyan eq. (3.15)"
