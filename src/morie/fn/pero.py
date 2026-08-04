# morie.fn -- slice s03 (rootcoder007/morie)
"""Prioritized experience replay.

Source consulted (FETCHED): Schaul, T., Quan, J., Antonoglou, I. and
Silver, D. (2016).  Prioritized experience replay.  *ICLR*
(arXiv:1511.05952).  Section 3.3 gives the sampling distribution

    P(i) = p_i^alpha / sum_k p_k^alpha                          (eq. 1)

with two priority definitions: proportional, p_i = |delta_i| + epsilon,
and rank-based, p_i = 1 / rank(i) where rank orders by |delta|.
Section 3.4 gives the bias correction

    w_i = ( (1/N) (1/P(i)) )^beta                               (eq. 2)

and states, verbatim: "For stability reasons, we always normalize
weights by 1/max_i w_i so that they only scale the update downwards."
Algorithm 1 writes the same weight as w_j = (N P(j))^(-beta) / max_i w_i.

The paper's own annealing of beta from beta_0 to 1 over training is
included as ``beta_schedule``: beta_t = beta_0 + (1 - beta_0) t / T.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["prioritized_experience_replay"]


def prioritized_experience_replay(buffer, alpha=0.6, beta=0.4, eps=1e-6,
                                  variant="proportional", t=None, T=None,
                                  n_sample=None):
    """Sampling distribution, importance weights, and a deterministic draw.

    Parameters
    ----------
    buffer : array-like
        The TD errors delta_i, one per stored transition.
    alpha : float
        Prioritisation exponent; the paper uses 0.6 (proportional) and
        0.7 (rank-based) for its DQN experiments.
    beta : float
        Importance-sampling exponent at time ``t``.
    eps : float
        The additive constant of the proportional variant.
    variant : {"proportional", "rank"}
        Priority definition.
    t, T : float, optional
        Anneal beta linearly from ``beta`` at t = 0 to 1 at t = T.
    n_sample : int, optional
        Draw this many indices.  The draw is deterministic: the inverse
        CDF of P is evaluated at van der Corput points, never at a
        pseudo-random stream.

    Returns
    -------
    RichResult with payload:
        estimate : P(0)
        prob, weight, priority
        beta_t   : the annealed beta actually used
        sample   : the drawn indices (empty when n_sample is None)
    """
    d = k.vec(buffer)
    n = len(d)
    if variant == "rank":
        order = sorted(range(n), key=lambda i: (-abs(d[i]), i))
        p = [0.0] * n
        for rank, i in enumerate(order):
            p[i] = 1.0 / (rank + 1.0)
    else:
        p = [abs(x) + float(eps) for x in d]
    pa = [x ** float(alpha) for x in p]
    tot = 0.0
    for x in pa:
        tot += x
    prob = [x / tot if tot > 0.0 else 0.0 for x in pa]
    b = float(beta)
    if t is not None and T is not None and float(T) > 0.0:
        b = b + (1.0 - b) * (float(t) / float(T))
        if b > 1.0:
            b = 1.0
    w = [(n * q) ** (-b) if q > 0.0 else 0.0 for q in prob]
    mx = 0.0
    for x in w:
        if x > mx:
            mx = x
    w = [x / mx if mx > 0.0 else 0.0 for x in w]
    sample = []
    if n_sample is not None:
        cum = []
        c = 0.0
        for q in prob:
            c += q
            cum.append(c)
        for j in range(int(n_sample)):
            u = k.vdc(j, 2)
            idx = n - 1
            for i in range(n):
                if u < cum[i]:
                    idx = i
                    break
            sample.append(idx)
    return RichResult(
        title="Prioritized experience replay",
        summary_lines=[("entries", n), ("beta", b)],
        payload={
            "estimate": prob[0] if prob else float("nan"),
            "prob": prob,
            "weight": w,
            "priority": p,
            "beta_t": b,
            "sample": sample,
            "n": n,
            "method": "Prioritized experience replay (Schaul et al. 2016, eqs. 1-2)",
        },
    )


def cheatsheet():
    return "pero: Prioritized experience replay"
