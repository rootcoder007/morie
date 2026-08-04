"""Bootstrap variance of the sample mean."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult
from .btmult import _counts

__all__ = ["boot_var_mean"]


def boot_var_mean(x, B=200, rng=2, exhaustive=False):
    """Var* = (1/B) sum_b (xbar*_b - mean_b xbar*_b)^2.

    Efron, B. (1979), "Bootstrap methods: another look at the
    jackknife", *The Annals of Statistics* 7(1), 1-26,
    doi:10.1214/aos/1176344552, p. 3, steps 1-3 and Eq. (2.8), read
    from the Project Euclid PDF rendered as page images.  For the
    simplest case, F putting all its mass on 0 and 1, Efron prints
    E*(xbar* - xbar) = 0 and Var*(xbar* - xbar) = xbar(1 - xbar)/n;
    that is the complete-enumeration answer this module reproduces
    exactly when ``exhaustive`` is set, and it is the anchor the
    module is checked against.

    More generally the complete bootstrap variance of the mean is
    sigma-hat^2/n with sigma-hat^2 = sum (x_i - xbar)^2 / n, the
    population form -- which on 0/1 data is xbar(1 - xbar), so Eq. (2.8)
    is the special case.

    Resampling is deterministic; see ``boot_multinomial_weights`` for
    the van der Corput construction and the meaning of ``rng``.

    Parameters
    ----------
    x : array-like
        The sample.
    B : int
        Replications when not enumerating.
    rng : int
        Base of the van der Corput sequence.
    exhaustive : bool
        Enumerate all n^n resamples (n <= 6), giving the complete
        bootstrap distribution rather than a sample from it.

    Returns
    -------
    estimate : the bootstrap variance, same as var_b
    var_b    : (1/B) sum (xbar*_b - mean of them)^2
    mean_b   : the B bootstrap means
    """
    v = core.vec(x)
    n = len(v)
    if n == 0:
        raise ValueError("boot_var_mean: x is empty")
    B = int(B)
    if not exhaustive and B < 1:
        raise ValueError("boot_var_mean: B must be at least 1")
    rng = int(rng)
    if rng < 2:
        raise ValueError("boot_var_mean: rng must be a base of at least 2")
    cs = _counts(n, B, rng, bool(exhaustive))
    mb = []
    for row in cs:
        s = 0.0
        for i in range(n):
            s += row[i] * v[i]
        mb.append(s / n)
    nb = len(mb)
    mm = 0.0
    for e in mb:
        mm += e
    mm /= nb
    vv = 0.0
    for e in mb:
        vv += (e - mm) * (e - mm)
    vv /= nb
    return RichResult(payload={
        "estimate": vv,
        "var_b": vv,
        "mean_b": mb,
        "grand_mean": mm,
        "B": nb,
        "n": n,
        "exhaustive": bool(exhaustive),
        "method": "Bootstrap variance of the sample mean",
    })


def cheatsheet():
    return "btvarm: Bootstrap variance of the sample mean"
