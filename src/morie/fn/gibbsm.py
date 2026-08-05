# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Gibbs sampler.

Geman and Geman (1984), "Stochastic relaxation, Gibbs distributions,
and the Bayesian restoration of images", IEEE Trans. Pattern Analysis
and Machine Intelligence 6(6):721-741, doi:10.1109/TPAMI.1984.4767596.
One sweep replaces each block in turn by a draw from its full
conditional,

    x_i ~ p(x_i | x_{-i}),

leaving the joint distribution invariant.  Each conditional is
supplied as a function of the current state and one uniform variate,
so the sampler is an inverse-CDF draw; the uniforms come from the van
der Corput sequence, which makes the whole run reproducible and
identical across language arms.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["gibbs_sampler"]


def gibbs_sampler(conditionals, x0, n_iter=100, burn=0):
    """Run a Gibbs sweep n_iter times from x0.

    Parameters
    ----------
    conditionals : sequence of callables
        conditionals[i](x, u) returns the new value of block i given
        the current state x and a uniform variate u in (0, 1).
    x0 : array-like
        Starting state, one entry per conditional.
    n_iter : int
        Number of sweeps.
    burn : int
        Sweeps discarded before the summaries are formed.
    """
    x = core.vec(x0)
    d = len(x)
    if d == 0:
        raise ValueError("gibbs_sampler: x0 is empty")
    cs = list(conditionals)
    if len(cs) != d:
        raise ValueError("gibbs_sampler: one conditional per block is required")
    for c in cs:
        if not callable(c):
            raise ValueError("gibbs_sampler: every conditional must be callable")
    it = int(n_iter)
    if it < 1:
        raise ValueError("gibbs_sampler: n_iter must be at least 1")
    bn = int(burn)
    if bn < 0 or bn >= it:
        raise ValueError("gibbs_sampler: burn must lie in [0, n_iter)")
    counter = 0
    draws = []
    for _ in range(it):
        for i in range(d):
            u = core.vdc(counter + 1)
            counter += 1
            x[i] = float(cs[i](list(x), u))
        draws.append(list(x))
    kept = draws[bn:]
    means = [sum(row[j] for row in kept) / len(kept) for j in range(d)]
    return RichResult(
        title="Gibbs sampler",
        summary_lines=[("sweeps", it), ("blocks", d)],
        payload={
            "estimate": means[0],
            "mean": means,
            "draws": draws,
            "last": list(x),
            "n": it,
            "method": "componentwise full-conditional updates with van der Corput uniforms, Geman & Geman (1984)",
        },
    )


def cheatsheet():
    return "gibbsm: Gibbs sampler"


# compact alias per ledger/NAMING.md
gibbssampler = gibbs_sampler
