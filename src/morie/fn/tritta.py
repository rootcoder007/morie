# morie.fn -- function file (rootcoder007/morie)
"""Triangular gated self-attention over the AlphaFold pair representation."""

from __future__ import annotations

import math

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_triangle_attn"]


def alphafold_triangle_attn(z, wq, wk, wv, wb, wg, wo, mode="starting"):
    """Triangular gated self-attention -- Algorithms 13 and 14, pp. 19-20.

    Edge ``ij`` attends over the edges sharing its starting node ``i``
    (Algorithm 13) or its ending node ``j`` (Algorithm 14).  The attention
    logit is a scaled query-key product modulated by a bias read off the
    third edge of the triangle, which is what makes the operation
    triangular rather than ordinary attention.

    Every weight is supplied by the caller.

    Parameters
    ----------
    z : list of list of list of float
        Pair representation, ``n x n x cz``.
    wq, wk, wv : list of list of list of float
        Per-head query, key and value projections; ``wq[h]`` is ``c x cz``
        (line 2, no bias).
    wb : list of list of float
        Per-head bias projection, ``nhead x cz``; ``wb[h]`` maps an edge to
        the scalar ``b^h`` of line 3.
    wg : list of list of list of float
        Per-head gate projections; ``wg[h]`` is ``c x cz`` (line 4).
    wo : list of list of float
        Output projection, ``cz x (nhead * c)`` (line 7).
    mode : {"starting", "ending"}
        Which of Algorithm 13 / Algorithm 14 to apply.

    Returns
    -------
    result : RichResult
        Keys: ``z`` (``z~``), ``attn`` (the attention tensor
        ``nhead x n x n x n``), ``estimate``, ``n``, ``method``.

    Notes
    -----
    Each attention distribution sums to one over its last index; the parity
    harness checks that directly rather than trusting agreement between the
    two arms.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithms 13-14
    """
    if mode not in ("starting", "ending"):
        raise ValueError("mode must be 'starting' or 'ending'")
    n = len(z)
    cz = len(z[0][0])
    nh = len(wq)
    c = len(wq[0])
    scale = 1.0 / math.sqrt(c)

    zn = [[A.lnorm(z[i][j]) for j in range(n)] for i in range(n)]
    q = [[[A.lin(zn[i][j], wq[h]) for j in range(n)] for i in range(n)]
         for h in range(nh)]
    k = [[[A.lin(zn[i][j], wk[h]) for j in range(n)] for i in range(n)]
         for h in range(nh)]
    v = [[[A.lin(zn[i][j], wv[h]) for j in range(n)] for i in range(n)]
         for h in range(nh)]
    b = [[[A.vdot(wb[h], zn[i][j]) for j in range(n)] for i in range(n)]
         for h in range(nh)]
    g = [[[[A.sigm(x) for x in A.lin(zn[i][j], wg[h])] for j in range(n)]
          for i in range(n)] for h in range(nh)]

    attn = []
    o = []
    for h in range(nh):
        ah, oh = [], []
        for i in range(n):
            arow, orow = [], []
            for j in range(n):
                # line 5: the third edge of the triangle enters as a bias
                if mode == "starting":
                    logits = [scale * A.vdot(q[h][i][j], k[h][i][kk]) + b[h][j][kk]
                              for kk in range(n)]
                else:
                    logits = [scale * A.vdot(q[h][i][j], k[h][kk][j]) + b[h][kk][i]
                              for kk in range(n)]
                a = A.smax(logits)
                # line 6: gated weighted sum of values
                if mode == "starting":
                    ov = [sum(a[kk] * v[h][i][kk][t] for kk in range(n))
                          for t in range(c)]
                else:
                    ov = [sum(a[kk] * v[h][kk][j][t] for kk in range(n))
                          for t in range(c)]
                arow.append(a)
                orow.append([g[h][i][j][t] * ov[t] for t in range(c)])
            ah.append(arow)
            oh.append(orow)
        attn.append(ah)
        o.append(oh)

    # line 7: concatenate the heads and project back to cz
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            cat = []
            for h in range(nh):
                cat.extend(o[h][i][j])
            row.append(A.lin(cat, wo))
        out.append(row)

    flat = [out[i][j][t] for i in range(n) for j in range(n) for t in range(cz)]
    return RichResult(
        payload={
            "z": out,
            "attn": attn,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "mode": mode,
            "method": "AlphaFold triangular self-attention (%s node)" % mode,
        }
    )


def cheatsheet():
    return "tritta: triangular gated self-attention, starting and ending node"
