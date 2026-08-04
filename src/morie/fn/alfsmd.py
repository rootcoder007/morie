# morie.fn -- function file (rootcoder007/morie)
"""MSA row-wise and column-wise gated self-attention (AlphaFold)."""

from __future__ import annotations

import math

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_msa_attention"]


def alphafold_msa_attention(m, wq, wk, wv, wg, wo, z=None, wb=None,
                            mode="row"):
    """MSA gated self-attention -- Algorithms 7 and 8, pp. 15-16.

    The MSA stack alternates attention along the two axes.  The row-wise
    form (Algorithm 7) attends over residues within one sequence and adds a
    bias read off the pair representation, which is how the pair stack
    speaks back to the MSA stack.  The column-wise form (Algorithm 8)
    attends over sequences at a fixed residue and carries no pair bias.

    Every weight is supplied by the caller.

    Parameters
    ----------
    m : list of list of list of float
        MSA representation, ``s x n x cm``.
    wq, wk, wv, wg : list of list of list of float
        Per-head projections; ``wq[h]`` is ``c x cm``.  ``wg`` feeds the
        sigmoid gate of line 4.
    wo : list of list of float
        Output projection, ``cm x (nhead * c)``.
    z : list of list of list of float, optional
        Pair representation, ``n x n x cz``.  Required for ``mode="row"``.
    wb : list of list of float, optional
        Per-head pair-bias projection, ``nhead x cz`` (line 3 of
        Algorithm 7).  Required for ``mode="row"``.
    mode : {"row", "column"}
        Which of Algorithm 7 / Algorithm 8 to apply.

    Returns
    -------
    result : RichResult
        Keys: ``m`` (the update ``m~``), ``attn``, ``estimate``, ``n``,
        ``s``, ``method``.

    Notes
    -----
    Each attention distribution sums to one; the parity harness checks that
    property directly.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithms 7-8
    """
    if mode not in ("row", "column"):
        raise ValueError("mode must be 'row' or 'column'")
    if mode == "row" and (z is None or wb is None):
        raise ValueError("mode='row' needs the pair representation z and wb")
    s = len(m)
    n = len(m[0])
    nh = len(wq)
    c = len(wq[0])
    scale = 1.0 / math.sqrt(c)

    mn = [[A.lnorm(m[si][i]) for i in range(n)] for si in range(s)]
    q = [[[A.lin(mn[si][i], wq[h]) for i in range(n)] for si in range(s)]
         for h in range(nh)]
    k = [[[A.lin(mn[si][i], wk[h]) for i in range(n)] for si in range(s)]
         for h in range(nh)]
    v = [[[A.lin(mn[si][i], wv[h]) for i in range(n)] for si in range(s)]
         for h in range(nh)]
    g = [[[[A.sigm(x) for x in A.lin(mn[si][i], wg[h])] for i in range(n)]
          for si in range(s)] for h in range(nh)]
    if mode == "row":
        zn = [[A.lnorm(z[i][j]) for j in range(n)] for i in range(n)]
        bias = [[[A.vdot(wb[h], zn[i][j]) for j in range(n)] for i in range(n)]
                for h in range(nh)]

    attn, o = [], []
    for h in range(nh):
        ah, oh = [], []
        for si in range(s):
            arow, orow = [], []
            for i in range(n):
                if mode == "row":
                    logits = [scale * A.vdot(q[h][si][i], k[h][si][j]) + bias[h][i][j]
                              for j in range(n)]
                    a = A.smax(logits)
                    ov = [sum(a[j] * v[h][si][j][t] for j in range(n))
                          for t in range(c)]
                else:
                    logits = [scale * A.vdot(q[h][si][i], k[h][t2][i])
                              for t2 in range(s)]
                    a = A.smax(logits)
                    ov = [sum(a[t2] * v[h][t2][i][t] for t2 in range(s))
                          for t in range(c)]
                arow.append(a)
                orow.append([g[h][si][i][t] * ov[t] for t in range(c)])
            ah.append(arow)
            oh.append(orow)
        attn.append(ah)
        o.append(oh)

    out = []
    for si in range(s):
        row = []
        for i in range(n):
            cat = []
            for h in range(nh):
                cat.extend(o[h][si][i])
            row.append(A.lin(cat, wo))
        out.append(row)

    cm = len(out[0][0])
    flat = [out[si][i][t] for si in range(s) for i in range(n)
            for t in range(cm)]
    return RichResult(
        payload={
            "m": out,
            "attn": attn,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "s": s,
            "mode": mode,
            "method": "AlphaFold MSA gated self-attention (%s-wise)" % mode,
        }
    )


def cheatsheet():
    return "alfsmd: MSA row-wise (with pair bias) and column-wise attention"
