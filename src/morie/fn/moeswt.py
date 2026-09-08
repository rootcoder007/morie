# morie.fn -- slice s03 (rootcoder007/morie)
"""Switch Transformer top-1 routing with a capacity factor.

Source consulted (FETCHED): Fedus, W., Zoph, B. and Shazeer, N. (2022).
Switch transformers: scaling to trillion parameter models with simple
and efficient sparsity.  *JMLR* 23(120), 1-39 (arXiv:2101.03961).  The
paper states, verbatim, that "each token is routed to the expert with
the highest router probability, but each expert has a fixed batch size
of (total_tokens / num_experts) x capacity_factor.  If the tokens are
unevenly dispatched then certain experts will overflow ... resulting in
these tokens not being processed by this layer."  Overflowed tokens pass
through the residual connection unchanged, which is exactly what is done
here -- they are not silently reassigned.

The auxiliary load-balancing loss is the paper's equation (4),

    loss = alpha . N . sum_(i=1)^N f_i . P_i

with f_i the fraction of tokens dispatched to expert i (equation 5) and
P_i the mean router probability for expert i (equation 6); alpha = 1e-2
in the paper.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["moe_switch_routing"]


def moe_switch_routing(y, x=None, W_g=None, experts=None, capacity=1.25,
                       alpha=1e-2):
    """Top-1 route a batch of tokens under a capacity constraint.

    Parameters
    ----------
    y : 2-D array-like
        The token batch, one row per token.  (First slot, for signature
        stability.)
    x : 2-D array-like, optional
        The token batch; wins over ``y``.
    W_g : 2-D array-like
        Router weights, rows indexed by feature, columns by expert.
    experts : 2-D array-like or list of callable, optional
        Expert outputs; the identity when omitted.
    capacity : float
        The capacity factor.
    alpha : float
        Coefficient of the auxiliary load-balancing loss.

    Returns
    -------
    RichResult with payload:
        estimate    : the auxiliary loss
        aux_loss    : same as estimate
        assign      : expert index per token, -1 when dropped
        dropped     : number of tokens that overflowed
        f, P        : the two vectors of equation (4)
        expert_capacity
        out         : the layer output, one row per token
    """
    toks = k.mat(x if x is not None else y)
    T = len(toks)
    Wg = k.mat(W_g)
    N = len(Wg[0]) if Wg else 0
    cap = int(float(capacity) * T / N) if N else 0
    probs = []
    for t in range(T):
        probs.append(k.softmax(k.matvec(k.tr(Wg), toks[t])))
    used = [0] * N
    assign = [-1] * T
    for t in range(T):
        best = 0
        for i in range(1, N):
            if probs[t][i] > probs[t][best]:
                best = i
        if used[best] < cap:
            assign[t] = best
            used[best] += 1
    f = [used[i] / T if T else 0.0 for i in range(N)]
    P = [0.0] * N
    for t in range(T):
        for i in range(N):
            P[i] += probs[t][i] / T
    s = 0.0
    for i in range(N):
        s += f[i] * P[i]
    aux = float(alpha) * N * s
    out = []
    for t in range(T):
        a = assign[t]
        if a < 0:
            out.append(list(toks[t]))
        elif experts is None:
            out.append([probs[t][a] * z for z in toks[t]])
        elif callable(experts[a]):
            e = k.vec(experts[a](toks[t]))
            out.append([probs[t][a] * z for z in e])
        else:
            e = k.vec(experts[a])
            out.append([probs[t][a] * z for z in e])
    drop = 0
    for a in assign:
        if a < 0:
            drop += 1
    return RichResult(
        title="Switch Transformer routing",
        summary_lines=[("experts", N), ("dropped", drop)],
        payload={
            "estimate": aux,
            "aux_loss": aux,
            "assign": assign,
            "dropped": drop,
            "f": f,
            "P": P,
            "expert_capacity": cap,
            "out": out,
            "method": "Switch Transformer top-1 routing with capacity factor (Fedus et al. 2022, eqs. 4-6)",
        },
    )


def cheatsheet():
    return "moeswt: Switch Transformer top-1 routing with capacity factor"
