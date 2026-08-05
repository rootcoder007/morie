# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Jensen-Shannon divergence.

Lin (1991), "Divergence measures based on the Shannon entropy", IEEE
Transactions on Information Theory 37(1):145-151,
doi:10.1109/18.61115, equation (3.6): with M = (P + Q)/2,

    JS(P, Q) = H(M) - (H(P) + H(Q)) / 2.

It is symmetric, always finite (unlike the Kullback-Leibler
divergence), bounded above by log 2, and equal to log 2 exactly when P
and Q have disjoint support -- all four of which the tests check.  Its
square root is a metric (Endres and Schindelin 2003, IEEE Trans. Inf.
Theory 49(7):1858-1860, doi:10.1109/TIT.2003.813506).

The stub this module replaces carried the label "Jensen-Zhang (1986)",
which does not correspond to any traceable paper on this divergence;
the attribution above is Lin's, verified against the DOI.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["jenson_zhang_disparity"]


def _entropy(p):
    h = 0.0
    for v in p:
        if v > 0:
            h -= v * math.log(v)
    return h


def jenson_zhang_disparity(y, p, q):
    """Jensen-Shannon divergence between two distributions over the same support.

    Parameters
    ----------
    y : array-like
        Support labels; used only for its length, which must match p and q.
    p, q : array-like
        Non-negative weights; each is normalised to sum to one.
    """
    pv = core.vec(p)
    qv = core.vec(q)
    if len(pv) == 0:
        raise ValueError("jenson_zhang_disparity: p is empty")
    if len(qv) != len(pv):
        raise ValueError("jenson_zhang_disparity: p and q have different lengths")
    if y is not None:
        yv = core.vec(y)
        if len(yv) != len(pv):
            raise ValueError("jenson_zhang_disparity: y and p have different lengths")
    for v in pv + qv:
        if v < 0:
            raise ValueError("jenson_zhang_disparity: weights must be non-negative")
    sp = sum(pv)
    sq = sum(qv)
    if sp <= 0 or sq <= 0:
        raise ValueError("jenson_zhang_disparity: weights must sum to something positive")
    P = [v / sp for v in pv]
    Q = [v / sq for v in qv]
    M = [(P[i] + Q[i]) / 2.0 for i in range(len(P))]
    hp = _entropy(P)
    hq = _entropy(Q)
    hm = _entropy(M)
    js = hm - (hp + hq) / 2.0
    if js < 0:
        js = 0.0
    return RichResult(
        title="Jensen-Shannon divergence",
        summary_lines=[("support", len(P)), ("divergence", js)],
        payload={
            "estimate": js,
            "divergence": js,
            "distance": math.sqrt(js),
            "entropy_p": hp,
            "entropy_q": hq,
            "entropy_m": hm,
            "n": len(P),
            "method": "JS = H(M) - (H(P) + H(Q))/2 with M = (P+Q)/2, Lin (1991) eq. (3.6)",
        },
    )


def cheatsheet():
    return "jzdiff: Jensen-Shannon divergence"
