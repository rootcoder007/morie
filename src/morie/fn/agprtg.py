# morie.fn -- slice s03 (rootcoder007/morie)
"""Priority targets for a prioritized replay buffer.

Source consulted (FETCHED): Schaul, T., Quan, J., Antonoglou, I. and
Silver, D. (2016).  Prioritized experience replay.  *ICLR*
(arXiv:1511.05952), section 3.3 and algorithm 1:

    P(i) = p_i^alpha / sum_k p_k^alpha
    w_i  = (N P(i))^(-beta) / max_j w_j

with the proportional variant p_i = |delta_i| + epsilon and the
rank-based variant p_i = 1 / rank(i).  For AlphaZero-style training the
temporal-difference error delta is replaced by the value residual
|z - v|, which is the module's own formula line and is the natural
analogue: the value head's own error on the stored outcome.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_priority_target"]


def alphazero_priority_target(replay_buffer, priorities=None, z=None, v=None,
                              alpha=0.6, beta=0.4, eps=1e-6, variant="proportional"):
    """Sampling probabilities and importance weights from value residuals.

    Parameters
    ----------
    replay_buffer : array-like
        The buffer; only its length is used unless ``z``/``v`` are drawn
        from it (it may be a list of (z, v) pairs).
    priorities : array-like, optional
        Raw priorities p_i.  When absent they are built from |z - v|.
    z, v : array-like, optional
        Stored outcomes and value-head predictions.
    alpha : float
        Prioritisation exponent; 0 recovers uniform sampling.
    beta : float
        Importance-sampling exponent; 1 fully corrects the bias.
    eps : float
        The additive constant of the proportional variant.
    variant : {"proportional", "rank"}
        Which priority definition to use.

    Returns
    -------
    RichResult with payload:
        estimate : P(0), the sampling probability of the first entry
        prob     : P(i) for every entry
        weight   : the normalised importance weights w_i
        priority : the raw p_i
    """
    if priorities is not None:
        raw = k.vec(priorities)
    elif z is not None and v is not None:
        zz = k.vec(z)
        vv = k.vec(v)
        raw = [abs(zz[i] - vv[i]) for i in range(len(zz))]
    else:
        rows = k.mat(replay_buffer)
        raw = [abs(r[0] - r[1]) for r in rows]
    n = len(raw)
    if variant == "rank":
        order = sorted(range(n), key=lambda i: (-raw[i], i))
        p = [0.0] * n
        for rank, i in enumerate(order):
            p[i] = 1.0 / (rank + 1.0)
    else:
        p = [x + float(eps) for x in raw]
    pa = [x ** float(alpha) for x in p]
    tot = 0.0
    for x in pa:
        tot += x
    prob = [x / tot if tot > 0.0 else 0.0 for x in pa]
    w = [(n * q) ** (-float(beta)) if q > 0.0 else 0.0 for q in prob]
    mx = 0.0
    for x in w:
        if x > mx:
            mx = x
    w = [x / mx if mx > 0.0 else 0.0 for x in w]
    return RichResult(
        title="Prioritized replay targets",
        summary_lines=[("entries", n)],
        payload={
            "estimate": prob[0] if prob else float("nan"),
            "prob": prob,
            "weight": w,
            "priority": p,
            "n": n,
            "method": "Prioritized replay priorities from the value residual |z - v|",
        },
    )


def cheatsheet():
    return "agprtg: AlphaZero prioritized replay target"
