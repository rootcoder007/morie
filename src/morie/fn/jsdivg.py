# morie.fn -- function file (rootcoder007/morie)
"""Jensen-Shannon divergence."""

from __future__ import annotations

import math

from . import _t4core as T

from ._richresult import RichResult

__all__ = ["jensen_shannon_divergence"]


def _jsd(p, q, base):
    lg = math.log(base)
    s1 = 0.0
    s2 = 0.0
    for pi, qi in zip(p, q):
        m = pi + qi
        if pi != 0.0 and m != 0.0:
            s1 += pi * math.log(2.0 * pi / m) / lg
        if qi != 0.0 and m != 0.0:
            s2 += qi * math.log(2.0 * qi / m) / lg
    return 0.5 * (s1 + s2)


def jensen_shannon_divergence(p, q, base=2.0, normalize=True):
    """Jensen-Shannon divergence between two discrete distributions.

    Formula: with ``M = (P + Q)/2``,

        ``JSD(P, Q) = (1/2) KL(P || M) + (1/2) KL(Q || M)``
                    ``= (1/2) sum p log(2p/(p+q)) + (1/2) sum q log(2q/(p+q))``

    Unlike the Kullback-Leibler divergence it is symmetric, always
    finite, and bounded by ``log 2`` (i.e. by 1 in bits) -- the
    finiteness is the point of averaging into ``M``, since ``KL(P||Q)``
    is infinite wherever ``Q`` puts no mass and ``P`` does.  Terms with
    ``p = 0`` or ``q = 0`` contribute nothing, by the usual
    ``0 log 0 = 0`` convention.  Its square root is a metric, which is
    what makes it usable as a distance.

    Parameters
    ----------
    p, q : array-like
        Non-negative weights over the same support.
    base : float
        Logarithm base; 2 gives bits, ``math.e`` gives nats.
    normalize : bool
        Rescale ``p`` and ``q`` to sum to one first.

    Returns
    -------
    RichResult
        ``estimate``, ``distance`` (its square root), ``bound``
        (``log 2`` in the chosen base), ``base``, ``n``, ``method``.

    References
    ----------
    Lin (1991), Divergence measures based on the Shannon entropy, IEEE
    Transactions on Information Theory 37:145-151.  Paywalled at IEEE;
    the coded form was read from Drost's ``philentropy`` package,
    src/distances_internal.h::jensen_shannon_internal (tarball
    philentropy_0.10.0 fetched from CRAN), which computes
    ``0.5*(sum p log(2p/(p+q)) + sum q log(2q/(p+q)))`` with exactly the
    zero-guards used here.
    """
    p = T.vec(p)
    q = T.vec(q)
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    if any(v < 0 for v in p) or any(v < 0 for v in q):
        raise ValueError("p and q must be non-negative")
    if normalize:
        sp, sq = sum(p), sum(q)
        if sp <= 0 or sq <= 0:
            raise ValueError("p and q must have positive total mass")
        p = [v / sp for v in p]
        q = [v / sq for v in q]
    d = _jsd(p, q, base)
    if d < 0.0:
        d = 0.0
    return RichResult(
        payload={
            "estimate": float(d),
            "distance": float(math.sqrt(d)),
            "bound": float(math.log(2.0) / math.log(base)),
            "base": float(base),
            "n": int(len(p)),
            "method": "Jensen-Shannon divergence",
        }
    )


def cheatsheet():
    return "jensen_shannon_divergence(p, q, base=2): JSD = mean KL to the mixture."


# compact alias per ledger/NAMING.md
jsd = jensen_shannon_divergence
