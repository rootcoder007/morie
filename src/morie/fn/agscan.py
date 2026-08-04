# morie.fn -- slice s03 (rootcoder007/morie)
"""Self-consistency of a policy network across repeated runs.

There is no equation for this in the AlphaZero literature: Silver et al.
(2018), arXiv:1712.01815 (FETCHED), reports run-to-run variation only as
Elo curves, and states no self-consistency statistic.  Rather than
invent an attribution, this function computes the two quantities that
are actually well defined for a set of policies over the same action
space, and cites them to where they *are* defined:

    Shannon entropy      H(p) = - sum_a p_a log p_a
                         (Shannon 1948, *Bell System Technical Journal*
                         27, 379-423, eq. 11)
    Jensen-Shannon
    divergence           JSD(p_1..p_m) = H(pbar) - (1/m) sum_i H(p_i)
                         (Lin 1991, *IEEE Trans. Inf. Theory* 37(1),
                         145-151, eq. 3.1)

JSD is zero exactly when every run produced the same policy and is
bounded by log m, so it is the natural consistency score; the per-run
entropies are reported alongside because that is what the module's own
formula line asks for.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_self_consistency"]


def _entropy(p):
    h = 0.0
    for x in p:
        if x > 0.0:
            h -= x * math.log(x)
    return h


def alphazero_self_consistency(policy_net, seeds=None):
    """Entropy spread and Jensen-Shannon divergence across runs.

    Parameters
    ----------
    policy_net : callable or 2-D array-like
        Either the policies themselves, one row per run, or a callable
        ``seed -> p`` applied to every entry of ``seeds``.
    seeds : array-like, optional
        Run identifiers passed to ``policy_net`` when it is callable.
        These label runs; they are not fed to any generator here.

    Returns
    -------
    RichResult with payload:
        estimate    : the Jensen-Shannon divergence, in nats
        jsd         : same as estimate
        entropies   : H(p_i) per run
        mean_entropy, sd_entropy, range_entropy
        mean_policy : the mixture pbar
    """
    if callable(policy_net):
        rows = [k.vec(policy_net(s)) for s in (seeds if seeds is not None else [])]
    else:
        rows = k.mat(policy_net)
    m = len(rows)
    if m == 0:
        return RichResult(payload={"estimate": float("nan"), "jsd": float("nan"),
                                   "entropies": [], "n": 0,
                                   "method": "Policy self-consistency"})
    norm = []
    for p in rows:
        t = 0.0
        for x in p:
            t += x
        norm.append([x / t for x in p] if t > 0.0 else p)
    K = len(norm[0])
    pbar = [0.0] * K
    for p in norm:
        for a in range(K):
            pbar[a] += p[a] / m
    ent = [_entropy(p) for p in norm]
    jsd = _entropy(pbar) - k.mean(ent)
    return RichResult(
        title="Policy self-consistency across runs",
        summary_lines=[("JSD (nats)", jsd), ("runs", m)],
        payload={
            "estimate": jsd,
            "jsd": jsd,
            "entropies": ent,
            "mean_entropy": k.mean(ent),
            "sd_entropy": k.sd(ent, 1) if m > 1 else float("nan"),
            "range_entropy": max(ent) - min(ent),
            "mean_policy": pbar,
            "n": m,
            "method": ("Self-consistency by Shannon entropy (1948 eq. 11) and "
                       "Jensen-Shannon divergence (Lin 1991 eq. 3.1); the "
                       "AlphaZero papers state no such statistic"),
        },
    )


def cheatsheet():
    return "agscan: AlphaZero self-consistency check across re-runs"
