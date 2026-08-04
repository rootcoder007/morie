# morie.fn -- slice s03 (rootcoder007/morie)
"""Posterior stick-breaking weights.

Source consulted: Ishwaran, H. and James, L. F. (2001).  Gibbs sampling
methods for stick-breaking priors.  *Journal of the American Statistical
Association* 96(453), 161-173.  Given a partition of n observations into
clusters with counts n_1, ..., n_K in the *order of the sticks*, the
conditional posterior of each stick fraction is

    V_k | data ~ Beta( 1 + n_k , alpha + sum_(j > k) n_j )

which is the conjugate update of the Beta(1, alpha) prior of Sethuraman
(1994), and the weights follow by the same product as before.  The 2001
JASA paper is paywalled; the conditional is quoted in its standard
published form.

The posterior *mean* stick fraction is used, not a draw, so the result
is a point summary rather than one sample -- E[V_k] = (1 + n_k) /
(1 + n_k + alpha + sum_(j>k) n_j).  Nothing here consults a generator.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["stick_breaking_post"]


def stick_breaking_post(partition, alpha=1.0):
    """Posterior mean stick fractions and weights from cluster counts.

    Parameters
    ----------
    partition : array-like
        Either cluster labels per observation, or the counts themselves
        when every entry is a nonnegative integer count in stick order.
    alpha : float
        The DP concentration.

    Returns
    -------
    estimate : pi_1
    pi, V    : posterior mean weights and stick fractions
    counts   : the cluster counts used
    """
    v = k.vec(partition)
    ints = all(abs(x - round(x)) < 1e-12 and x >= 0.0 for x in v)
    labs = []
    for x in v:
        if x not in labs:
            labs.append(x)
    if ints and len(labs) == len(v) and len(v) > 1:
        counts = list(v)
    else:
        labs = sorted(set(v))
        counts = [float(sum(1 for x in v if x == c)) for c in labs]
    K = len(counts)
    a = float(alpha)
    Vs = []
    for i in range(K):
        tail = 0.0
        for j in range(i + 1, K):
            tail += counts[j]
        Vs.append((1.0 + counts[i]) / (1.0 + counts[i] + a + tail))
    pi = []
    rest = 1.0
    for i in range(K):
        pi.append(Vs[i] * rest)
        rest *= (1.0 - Vs[i])
    return RichResult(
        title="Posterior stick-breaking weights",
        summary_lines=[("clusters", K), ("alpha", a)],
        payload={
            "estimate": pi[0] if pi else float("nan"),
            "pi": pi,
            "V": Vs,
            "counts": counts,
            "remainder": rest,
            "method": "Ishwaran and James (2001) conjugate stick-breaking posterior, at its mean",
        },
    )


def cheatsheet():
    return "sbpst: Posterior stick-breaking weights"
