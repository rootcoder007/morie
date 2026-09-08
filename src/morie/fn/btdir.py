# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Dirichlet(1,...,1) weight matrix for the Bayesian bootstrap.

Rubin, D. B. (1981), "The Bayesian Bootstrap", *The Annals of
Statistics* 9(1), 130-134, doi:10.1214/aos/1176345338 (verified against
Crossref).  Section 2: the posterior of the multinomial probability
vector under the improper Dirichlet(0,...,0) prior is Dirichlet(1,...,1)
on the n observed values, and Rubin's stated simulation recipe is

    draw u_1,...,u_(n-1) iid U(0,1), sort them to
    0 = v_0 <= v_1 <= ... <= v_(n-1) <= v_n = 1,
    set w_i = v_i - v_(i-1).

The gaps between the order statistics of n-1 iid uniforms are exactly
Dirichlet(1,...,1); no rejection or gamma-ratio step is involved.  This
module is the weight generator only; ``btbayes`` consumes it.

Deterministic by construction: the uniforms come from the package's
shared Lehmer minstd stream (``_tail1core.Lcg`` / ``.t1_lcg``), whose
intermediates fit exactly in a float64 so the Python and R arms produce
bit-identical draws.
"""

from __future__ import annotations

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["boot_dirichlet_weights"]


def dirichlet_rows(n, B, seed=1):
    """The B x n weight matrix, as a list of lists.  Shared with btbayes."""
    n = int(n)
    B = int(B)
    if n < 1:
        raise ValueError("boot_dirichlet_weights: n must be at least 1")
    if B < 1:
        raise ValueError("boot_dirichlet_weights: B must be at least 1")
    g = C.Lcg(seed)
    W = []
    for _ in range(B):
        if n == 1:
            W.append([1.0])
            continue
        v = sorted(g.unif() for _ in range(n - 1))
        row = [v[0]]
        for i in range(1, n - 1):
            row.append(v[i] - v[i - 1])
        row.append(1.0 - v[n - 2])
        W.append(row)
    return W


def boot_dirichlet_weights(n, B=200, rng=1):
    """Draw B Dirichlet(1,...,1) weight vectors of length n.

    Parameters
    ----------
    n : int
        Sample size; the weight vectors have this length.
    B : int
        Number of weight vectors.
    rng : int
        Seed for the shared deterministic stream.

    Returns
    -------
    RichResult
        ``W`` (B x n), ``rowsum_max_err`` (departure of the row sums from
        1), ``w_mean`` (grand mean, converges to 1/n), ``w_var`` (grand
        variance, converges to (n-1)/(n^2 (n+1))), ``n``, ``B``.
    """
    W = dirichlet_rows(n, B, rng)
    n = int(n)
    B = int(B)
    err = 0.0
    tot = 0.0
    tot2 = 0.0
    for row in W:
        s = 0.0
        for w in row:
            s += w
            tot += w
            tot2 += w * w
        d = abs(s - 1.0)
        if d > err:
            err = d
    N = n * B
    m = tot / N
    return RichResult(
        title="Dirichlet(1,...,1) bootstrap weights",
        summary_lines=[("n", n), ("B", B), ("w_mean", m)],
        payload={
            "W": W,
            "rowsum_max_err": err,
            "w_mean": m,
            "w_var": tot2 / N - m * m,
            "w_min": min(min(r) for r in W),
            "w_max": max(max(r) for r in W),
            "n": n,
            "B": B,
            "estimate": m,
            "method": "Rubin (1981) Ann. Statist. 9(1):130-134, uniform-gap Dirichlet(1,...,1)",
        },
    )


def cheatsheet():
    return "btdir: Dirichlet(1,..,1) weights as gaps of n-1 sorted uniforms (Rubin 1981)"
