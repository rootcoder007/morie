# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""ALiBi: attention with a linear positional bias.

Source: Press, O., Smith, N. A. and Lewis, M. (2022), "Train short, test
long: attention with linear biases enables input length extrapolation",
ICLR 2022, arXiv:2108.12409, read from the fetched PDF.  Page 4 gives the
modification verbatim -- it is applied *after* the query-key dot product:

    softmax( q_i K^T + m * [-(i-1), ..., -2, -1, 0] )

and states the slope schedule: "for n heads, our set of slopes is the
geometric sequence that starts at 2^(-8/n) and uses that same value as
its ratio", so head k (1-based) gets m_k = 2^(-8k/n).

This module carries the whole ALiBi implementation for the shelf; the
sibling module ``alibi`` delegates to the bias builder here rather than
holding a second copy.

The bias used is -m|i - j|, symmetric in the distance.  On the causal
lower triangle j <= i that is identical to the paper's
[-(i-1), ..., -1, 0] row, because |i - j| = i - j there; the symmetric
form simply extends it to the non-causal case.  Set ``causal=True`` to
mask the future out entirely, which reproduces the paper exactly.

No position embeddings are added anywhere: that absence is the method.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["alibi_position_bias"]


def head_slopes(n_heads):
    """m_k = 2^(-8k/n) for k = 1..n, the paper's geometric schedule."""
    if int(n_heads) < 1:
        raise ValueError("alibi_position_bias: n_heads must be at least one")
    n = int(n_heads)
    return [2.0 ** (-8.0 * (k + 1.0) / n) for k in range(n)]


def alibi_bias(n_q, n_k, slope, causal=False):
    """The n_q by n_k matrix of -m|i - j| penalties (-inf where masked)."""
    out = []
    for i in range(n_q):
        row = []
        for j in range(n_k):
            if causal and j > i:
                row.append(float("-inf"))
            else:
                row.append(-float(slope) * abs(i - j))
        out.append(row)
    return out


def _softmax_row(v):
    mx = None
    for x in v:
        if x != float("-inf") and (mx is None or x > mx):
            mx = x
    if mx is None:
        raise ValueError("alibi_position_bias: a query row has every key masked out")
    e = [0.0 if x == float("-inf") else math.exp(x - mx) for x in v]
    t = 0.0
    for x in e:
        t += x
    return [x / t for x in e]


def alibi_position_bias(y=None, Q=None, K=None, V=None, slopes=None, causal=False):
    """Attention output under a linear positional bias.

    Parameters
    ----------
    y : ignored
        Accepted for interface compatibility with the rest of the shelf.
    Q : array-like
        n_q by d queries.
    K : array-like
        n_k by d keys.
    V : array-like
        n_k by d_v values.
    slopes : float or array-like, optional
        The head slope m, or one per head.  Defaults to a single head
        with the paper's schedule, m = 2^-8.
    causal : bool
        Mask keys after the query position, as in the paper.

    Returns
    -------
    output : n_q by d_v attention output (one head) or a list per head
    weights : the attention weights of the first head
    bias : the bias matrix of the first head
    """
    if Q is None or K is None or V is None:
        raise ValueError("alibi_position_bias: Q, K and V are all required")
    Qm = [[float(v) for v in r] for r in Q]
    Km = [[float(v) for v in r] for r in K]
    Vm = [[float(v) for v in r] for r in V]
    nq = len(Qm)
    nk = len(Km)
    if nq == 0 or nk == 0:
        raise ValueError("alibi_position_bias: Q and K must be non-empty")
    d = len(Qm[0])
    if any(len(r) != d for r in Qm) or any(len(r) != d for r in Km):
        raise ValueError("alibi_position_bias: Q and K must share the key dimension")
    if len(Vm) != nk:
        raise ValueError("alibi_position_bias: V must have one row per key")
    dv = len(Vm[0])
    if slopes is None:
        sl = [2.0 ** -8.0]
    elif hasattr(slopes, "__len__"):
        sl = [float(v) for v in slopes]
    else:
        sl = [float(slopes)]
    if not sl:
        raise ValueError("alibi_position_bias: slopes is empty")
    sc = 1.0 / math.sqrt(d)
    outs = []
    W0 = None
    B0 = None
    for h, m in enumerate(sl):
        B = alibi_bias(nq, nk, m, causal)
        O = []
        Wh = []
        for i in range(nq):
            row = []
            for j in range(nk):
                dot = 0.0
                for t in range(d):
                    dot += Qm[i][t] * Km[j][t]
                row.append(dot * sc + B[i][j])
            w = _softmax_row(row)
            Wh.append(w)
            o = [0.0] * dv
            for j in range(nk):
                for t in range(dv):
                    o[t] += w[j] * Vm[j][t]
            O.append(o)
        outs.append(O)
        if h == 0:
            W0 = Wh
            B0 = B
    return RichResult(
        title="ALiBi attention",
        summary_lines=[("heads", len(sl)), ("n_q", nq), ("n_k", nk)],
        payload={
            "output": outs[0] if len(sl) == 1 else outs,
            "estimate": outs[0][0][0],
            "weights": W0,
            "bias": B0,
            "slopes": sl,
            "n_q": nq,
            "n_k": nk,
            "d": d,
            "d_v": dv,
            "causal": bool(causal),
            "method": "softmax(QK'/sqrt(d) - m|i-j|) V; Press, Smith and Lewis (2022), arXiv:2108.12409",
        },
    )


def cheatsheet():
    return "atalib: ALiBi attention with linear positional bias"


# compact alias per ledger/NAMING.md
alibipositionbias = alibi_position_bias
