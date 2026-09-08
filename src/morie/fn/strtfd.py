# morie.fn -- function file (rootcoder007/morie)
"""Stratified sampling design: allocation and the variance it buys."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["stratdes", "stratified_design"]


def stratdes(Nh, Sh, n, Ch=None, kind="neyman"):
    """Allocate a stratified sample and report the variance it achieves.

    All three of Cochran's allocations are here because the comparison
    is the point: proportional ignores the stratum spreads, Neyman puts
    the sample where the variance is, and the cost-optimal rule pulls
    it back out of strata that are expensive to measure.  The achieved
    variance is computed for the allocation actually returned, so the
    gain is a number rather than a claim.

    Formula: proportional  n_h prop. N_h
             Neyman        n_h prop. N_h S_h
             cost-optimal  n_h prop. N_h S_h / sqrt(C_h)
             V(ybar_st) = sum_h W_h^2 (1 - f_h) S_h^2 / n_h

    Parameters
    ----------
    Nh : array-like
        Population size of each stratum.
    Sh : array-like
        Stratum standard deviations (advance estimates).
    n : int
        Total sample size, at least 1 per stratum.
    Ch : array-like, optional
        Cost per unit in each stratum; required for kind="cost".
    kind : {"prop", "neyman", "cost"}
        Allocation rule.

    Returns
    -------
    RichResult
        ``nh``, ``nh_exact``, ``weights``, ``variance``, ``se``,
        ``Wh``, ``N``, ``n``, ``L``, ``kind``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Sections 5.3-5.5:
    Theorem 5.3 for the variance, Corollary 2 for proportional
    allocation, and the Neyman/cost-optimal rules n_h prop. N_h S_h and
    n_h prop. N_h S_h/sqrt(c_h) under the linear cost function (5.17).
    Chapter 5 read from the scanned original.  Cross-checked against
    the reference implementation in the CRAN package ``samplingbook``
    1.2.4, whose ``stratasamp`` uses exactly these three weight rules.
    """
    Nh = C.vec(Nh)
    Sh = C.vec(Sh)
    n = int(n)
    L = len(Nh)
    if len(Sh) != L:
        raise ValueError("Nh and Sh must have the same length")
    if any(v <= 0 for v in Nh):
        raise ValueError("stratum sizes must be positive")
    if any(v < 0 for v in Sh):
        raise ValueError("stratum standard deviations must be non-negative")
    if n < L:
        raise ValueError("n must be at least the number of strata")
    kind = str(kind).lower()
    if kind == "prop":
        w = list(Nh)
    elif kind == "neyman":
        w = [Nh[i] * Sh[i] for i in range(L)]
    elif kind == "cost":
        if Ch is None:
            raise ValueError("kind='cost' needs the per-unit costs Ch")
        Cv = C.vec(Ch)
        if len(Cv) != L or any(v <= 0 for v in Cv):
            raise ValueError("Ch must be positive and of length L")
        w = [Nh[i] * Sh[i] / math.sqrt(Cv[i]) for i in range(L)]
    else:
        raise ValueError("kind must be 'prop', 'neyman' or 'cost'")
    sw = sum(w)
    if sw <= 0:
        raise ValueError("the allocation weights are all zero")
    w = [v / sw for v in w]
    exact = [n * v for v in w]
    base = [max(1, int(v)) for v in exact]
    # Largest-remainder apportionment, ties on the lowest index, with a
    # floor of one unit per stratum so no stratum is left unsampled.
    while sum(base) > n:
        i = max((i for i in range(L) if base[i] > 1),
                key=lambda i: (base[i] - exact[i], -i))
        base[i] -= 1
    order = sorted(range(L), key=lambda i: (-(exact[i] - base[i]), i))
    j = 0
    while sum(base) < n:
        base[order[j % L]] += 1
        j += 1
    N = sum(Nh)
    W = [v / N for v in Nh]
    var = sum(W[i] ** 2 * (1.0 - base[i] / Nh[i]) * Sh[i] ** 2 / base[i]
              for i in range(L))
    return RichResult(payload={
        "nh": base, "nh_exact": exact, "weights": w, "variance": var,
        "se": math.sqrt(var), "Wh": W, "N": N, "n": n, "L": L,
        "kind": {"prop": 1.0, "neyman": 2.0, "cost": 3.0}[kind],
        "method": "Stratified allocation and achieved variance"})


stratified_design = stratdes


def cheatsheet():
    return "strtfd: n_h prop N_h / N_h S_h / N_h S_h/sqrt(C_h); V = sum W_h^2(1-f_h)S_h^2/n_h"
