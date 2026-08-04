# morie.fn -- function file (rootcoder007/morie)
"""Mean of a sum of random processes (Rangayyan eq. 3.13)."""


from math import fsum

from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["meansum", "rangayyan_ch3_mean_of_sum"]


def meansum(*processes):
    """Mean of a sum of random processes.

    Rangayyan (2024) eq. (3.13):  E[y] = mu_y = mu_x + mu_eta.

    Linearity of expectation needs no independence assumption, which is
    exactly what distinguishes eq. (3.13) from eq. (3.14) -- the variance
    identity that does.  Accepts any number of processes; each may be a
    sequence of samples or an already-computed mean.
    """
    if not processes:
        raise ValueError("need at least one process")
    means = []
    for p in processes:
        vals = aslist(p)
        if not vals:
            raise ValueError("every process needs at least one sample")
        means.append(fsum(vals) / len(vals))
    return RichResult(payload={
        "mean": fsum(means), "component_means": means,
        "n_processes": len(means),
        "method": "Rangayyan (2024) eq. (3.13)"})


rangayyan_ch3_mean_of_sum = meansum  # pre-policy spelling


def cheatsheet():
    return "rng013: mean of a sum, Rangayyan eq. (3.13)"
