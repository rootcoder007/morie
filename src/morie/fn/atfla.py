# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""FlashAttention: IO-aware block-tiled exact attention.

Source: Dao, T., Fu, D. Y., Ermon, S., Rudra, A. and Re, C. (2022),
"FlashAttention: fast and memory-efficient exact attention with
IO-awareness", NeurIPS 2022, arXiv:2205.14135, read from the fetched
PDF; and Dao, T. (2023), "FlashAttention-2", arXiv:2307.08691.

The claim in the title is the one that matters here: *exact*.
FlashAttention is not an approximation.  It never materialises the
n_q by n_k score matrix; it walks the keys in blocks and keeps a running
maximum m and a running denominator l, rescaling the accumulated output
whenever a new block raises the maximum:

    m_new = max(m_old, rowmax(S_block))
    l_new = e^(m_old - m_new) l_old + rowsum( e^(S_block - m_new) )
    O_new = e^(m_old - m_new) O_old + e^(S_block - m_new) V_block

with O divided by l once at the end.  The rescaling factor
e^(m_old - m_new) is what makes the recurrence exact rather than merely
stable: dropping it leaves earlier blocks normalised against a stale
maximum, and the error is small enough to look like rounding while being
systematic.

That exactness is the anchor, and it is a strong one: the output must
equal unblocked softmax attention to floating-point precision *for every
block size*, including block_size = 1 and block_size >= n_k.  A
tiling bug that is invisible at one block size shows up at another.

The IO savings this buys are a property of the memory hierarchy, not of
the arithmetic, so nothing here is faster than the naive version -- the
point of this implementation is that it computes the same number.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["flash_attention_block"]


def flash_attention_block(y=None, Q=None, K=None, V=None, block_size=2, causal=False):
    """Block-tiled attention with online softmax rescaling.

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
    block_size : int
        Number of keys per tile, at least one.
    causal : bool
        Mask keys after the query position.

    Returns
    -------
    output : n_q by d_v attention output
    l : the final softmax denominators
    m : the final running maxima
    n_blocks : how many key tiles were walked
    """
    if Q is None or K is None or V is None:
        raise ValueError("flash_attention_block: Q, K and V are all required")
    Qm = [[float(v) for v in r] for r in Q]
    Km = [[float(v) for v in r] for r in K]
    Vm = [[float(v) for v in r] for r in V]
    nq = len(Qm)
    nk = len(Km)
    if nq == 0 or nk == 0:
        raise ValueError("flash_attention_block: Q and K must be non-empty")
    d = len(Qm[0])
    if any(len(r) != d for r in Qm) or any(len(r) != d for r in Km):
        raise ValueError("flash_attention_block: Q and K must share the key dimension")
    if len(Vm) != nk:
        raise ValueError("flash_attention_block: V must have one row per key")
    dv = len(Vm[0])
    bs = int(block_size)
    if bs < 1:
        raise ValueError("flash_attention_block: block_size must be at least one")
    sc = 1.0 / math.sqrt(d)
    NEG = float("-inf")
    O = [[0.0] * dv for _ in range(nq)]
    l = [0.0] * nq
    m = [NEG] * nq
    nb = 0
    j0 = 0
    while j0 < nk:
        j1 = j0 + bs
        if j1 > nk:
            j1 = nk
        nb += 1
        for i in range(nq):
            row = []
            for j in range(j0, j1):
                if causal and j > i:
                    row.append(NEG)
                    continue
                dot = 0.0
                for t in range(d):
                    dot += Qm[i][t] * Km[j][t]
                row.append(dot * sc)
            bmax = NEG
            for v in row:
                if v > bmax:
                    bmax = v
            if bmax == NEG:
                continue
            mnew = m[i] if m[i] > bmax else bmax
            resc = 0.0 if m[i] == NEG else math.exp(m[i] - mnew)
            s = 0.0
            e = []
            for v in row:
                ev = 0.0 if v == NEG else math.exp(v - mnew)
                e.append(ev)
                s += ev
            l[i] = resc * l[i] + s
            for t in range(dv):
                acc = resc * O[i][t]
                for a, j in enumerate(range(j0, j1)):
                    acc += e[a] * Vm[j][t]
                O[i][t] = acc
            m[i] = mnew
        j0 = j1
    for i in range(nq):
        if l[i] <= 0.0:
            raise ValueError("flash_attention_block: a query row has every key masked out")
        for t in range(dv):
            O[i][t] = O[i][t] / l[i]
    return RichResult(
        title="FlashAttention (block-tiled, exact)",
        summary_lines=[("n_q", nq), ("n_k", nk), ("blocks", nb)],
        payload={
            "output": O,
            "estimate": O[0][0],
            "l": l,
            "m": m,
            "n_blocks": nb,
            "block_size": bs,
            "n_q": nq,
            "n_k": nk,
            "d": d,
            "d_v": dv,
            "causal": bool(causal),
            "method": "online-softmax block tiling, exact; Dao et al. (2022), arXiv:2205.14135",
        },
    )


def cheatsheet():
    return "atfla: FlashAttention IO-aware block-tiled attention"


# compact alias per ledger/NAMING.md
flashattentionblock = flash_attention_block
