"""Multinomial bootstrap resampling weights."""

from __future__ import annotations

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_multinomial_weights"]

MAXEX = 6


def _counts(n, B, rng, exhaustive):
    """Resample count vectors: one row per bootstrap replication."""
    if exhaustive:
        if n > MAXEX:
            raise ValueError("boot_multinomial_weights: exhaustive enumeration is capped at n = 6")
        rows = [[]]
        for _ in range(n):
            rows = [r + [j] for r in rows for j in range(n)]
        out = []
        for r in rows:
            cc = [0] * n
            for j in r:
                cc[j] += 1
            out.append(cc)
        return out
    out = []
    for b in range(B):
        cc = [0] * n
        for i in range(n):
            u = core.vdc(b * n + i, rng)
            j = int(u * n)
            if j >= n:
                j = n - 1
            cc[j] += 1
        out.append(cc)
    return out


def boot_multinomial_weights(n, B=200, rng=2, exhaustive=False):
    """The resampling weights P* = (N_1, ..., N_n)/n of the bootstrap.

    Efron, B. (1979), "Bootstrap methods: another look at the
    jackknife", *The Annals of Statistics* 7(1), 1-26,
    doi:10.1214/aos/1176344552, p. 3, steps 1-3, read from the Project
    Euclid PDF rendered as page images: construct F-hat putting mass
    1/n at each of x_1 ... x_n; draw X* of size n from F-hat with
    replacement (Eq. 2.4); approximate the distribution of R(X, F) by
    that of R(X*, F-hat) (Eq. 2.5).  A draw of size n with replacement
    from n points is a Multinomial(n; 1/n, ..., 1/n) count vector, so
    the weight vector is that count vector divided by n; every row of W
    sums to 1 and every entry is a multiple of 1/n.

    Determinism, which the parity check between the two language arms
    requires, replaces the random draw with the van der Corput
    low-discrepancy sequence already used elsewhere in this package:
    replication b, position i takes the point vdc(b n + i, rng), and
    ``rng`` is that sequence's base rather than a seed.  With
    ``exhaustive`` the sequence is not used at all: all n^n index
    tuples are enumerated, which is the complete bootstrap
    distribution rather than a sample from it, and B is ignored.

    Parameters
    ----------
    n : int
        Sample size.
    B : int
        Number of replications when not enumerating.
    rng : int
        Base of the van der Corput sequence, at least 2.
    exhaustive : bool
        Enumerate all n^n resamples instead (n <= 6).

    Returns
    -------
    estimate : the mean weight, which is 1/n
    W        : the B-by-n weight matrix
    counts   : the B-by-n multinomial count matrix
    """
    n = int(n)
    if n < 1:
        raise ValueError("boot_multinomial_weights: n must be at least 1")
    B = int(B)
    if not exhaustive and B < 1:
        raise ValueError("boot_multinomial_weights: B must be at least 1")
    rng = int(rng)
    if rng < 2:
        raise ValueError("boot_multinomial_weights: rng must be a base of at least 2")
    cs = _counts(n, B, rng, bool(exhaustive))
    W = [[c / n for c in row] for row in cs]
    tot = 0.0
    for row in W:
        for e in row:
            tot += e
    return RichResult(payload={
        "estimate": tot / (len(W) * n),
        "W": W,
        "counts": cs,
        "B": len(W),
        "n": n,
        "exhaustive": bool(exhaustive),
        "method": "Multinomial bootstrap weights",
    })


def cheatsheet():
    return "btmult: Multinomial bootstrap weights"
