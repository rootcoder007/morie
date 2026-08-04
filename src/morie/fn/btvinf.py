"""Empirical influence values by numerical perturbation."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult
from .infcrv import _resolve

__all__ = ["boot_influence_fn"]


def boot_influence_fn(x, stat="mean", eps=1e-3):
    """Empirical influence value at every sample point.

    The quotient is the one of Hampel, F. R. (1974), "The influence
    curve and its role in robust estimation", *Journal of the American
    Statistical Association* 69(346), 383-393,
    doi:10.1080/01621459.1974.10482962 (closed access, no open copy in
    any repository per Unpaywall), evaluated at finite eps at each
    observed point rather than in the limit:

        U_i = [T((1-eps) F_n + eps delta_{x_i}) - T(F_n)] / eps.

    These U_i are the empirical influence values the nonparametric
    delta method uses; ``estimate`` is that variance estimate,
    sum U_i^2 / n^2, which for the mean is exactly (n-1) s^2 / n^2,
    the usual variance of the sample mean with the population-variance
    denominator.

    T is a functional of a weighted sample, T(values, weights), or one
    of the names "mean", "var", "median"; see ``influence_function``.

    Parameters
    ----------
    x : array-like
        The sample.
    stat : callable or str
        The statistic whose influence values are wanted.
    eps : float
        Contamination weight.

    Returns
    -------
    estimate : the delta-method variance sum U_i^2 / n^2
    infl     : the n influence values U_i
    tf       : T(F_n)
    """
    T = _resolve(stat, "boot_influence_fn")
    v = core.vec(x)
    n = len(v)
    if n == 0:
        raise ValueError("boot_influence_fn: x is empty")
    e = float(eps)
    if not 0.0 < e < 1.0:
        raise ValueError("boot_influence_fn: eps must lie strictly between 0 and 1")
    base = T(v, [1.0 / n] * n)
    u = []
    ss = 0.0
    for i in range(n):
        vals = v + [v[i]]
        w = [(1.0 - e) / n] * n + [e]
        ui = (T(vals, w) - base) / e
        u.append(ui)
        ss += ui * ui
    return RichResult(payload={
        "estimate": ss / (n * n),
        "infl": u,
        "tf": base,
        "eps": e,
        "n": n,
        "method": "Influence function via numerical perturbation",
    })


def cheatsheet():
    return "btvinf: Influence function via numerical perturbation"
