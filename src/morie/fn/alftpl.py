# morie.fn -- function file (rootcoder007/morie)
"""Template pointwise attention of AlphaFold."""

from __future__ import annotations

import math

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_template_embed"]


def alphafold_template_embed(t, z, wq, wk, wv, wo):
    """Template pointwise attention -- Algorithm 17, p. 21.

    For each residue pair the pair representation queries the stack of
    template features at that same pair, and the templates are pooled by
    attention.  The attention runs over templates only, never across pairs,
    which is why it is called pointwise: pair ``ij`` never sees pair ``kl``.

    All weights are supplied by the caller.

    Parameters
    ----------
    t : list of list of list of list of float
        Template pair features, ``ntempl x n x n x ct``.
    z : list of list of list of float
        Pair representation, ``n x n x cz``.
    wq : list of list of list of float
        Per-head query projections from the pair representation;
        ``wq[h]`` is ``c x cz`` (line 1).
    wk, wv : list of list of list of float
        Per-head key and value projections from the template features;
        ``wk[h]`` is ``c x ct`` (line 2).
    wo : list of list of float
        Output projection, ``cz x (nhead * c)`` (line 5).

    Returns
    -------
    result : RichResult
        Keys: ``z`` (the update ``z~``), ``attn`` (``nhead x n x n x
        ntempl``), ``estimate``, ``n``, ``ntempl``, ``method``.

    Notes
    -----
    With a single template the attention is forced to one, so the pooled
    value is exactly that template's value projection.  The harness checks
    that closed form as well as the usual normalisation.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithm 17
    """
    nt = len(t)
    n = len(z)
    nh = len(wq)
    c = len(wq[0])
    scale = 1.0 / math.sqrt(c)

    attn, o = [], []
    for h in range(nh):
        ah, oh = [], []
        for i in range(n):
            arow, orow = [], []
            for j in range(n):
                q = A.lin(z[i][j], wq[h])
                logits = [scale * A.vdot(q, A.lin(t[st][i][j], wk[h]))
                          for st in range(nt)]
                a = A.smax(logits)
                vv = [A.lin(t[st][i][j], wv[h]) for st in range(nt)]
                arow.append(a)
                orow.append([sum(a[st] * vv[st][u] for st in range(nt))
                             for u in range(c)])
            ah.append(arow)
            oh.append(orow)
        attn.append(ah)
        o.append(oh)

    out = []
    for i in range(n):
        row = []
        for j in range(n):
            cat = []
            for h in range(nh):
                cat.extend(o[h][i][j])
            row.append(A.lin(cat, wo))
        out.append(row)

    cz = len(out[0][0])
    flat = [out[i][j][u] for i in range(n) for j in range(n) for u in range(cz)]
    return RichResult(
        payload={
            "z": out,
            "attn": attn,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "ntempl": nt,
            "method": "AlphaFold template pointwise attention",
        }
    )


def cheatsheet():
    return "alftpl: template pointwise attention pooling over templates"
