# morie.fn -- function file (rootcoder007/morie)
"""Invariant Point Attention (AlphaFold structure module)."""

from __future__ import annotations

import math

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_invariant_point"]


def alphafold_invariant_point(s, z, frames, wq, wk, wv, wqp, wkp, wvp, wb,
                              gamma, wo):
    """Invariant point attention -- Algorithm 22, p. 28.

    Attention over residues in which the logit carries, besides the usual
    query-key term and a pair bias, a squared distance between query and
    key points placed in each residue's own rigid frame.  Because that
    distance is measured between globally-placed points, and because the
    output points are mapped back through the inverse of the receiving
    frame, the whole operation is invariant under a global rigid motion of
    all frames.  The supplement proves this at equations (3)-(6); the
    parity harness checks it numerically.

    All weights, including the per-head scalar ``gamma``, are supplied by
    the caller.

    Parameters
    ----------
    s : list of list of float
        Single representation, ``n x cs``.
    z : list of list of list of float
        Pair representation, ``n x n x cz``.
    frames : list
        Backbone frames, one ``[R, t]`` per residue, with ``R`` a 3x3
        rotation given as a list of rows.
    wq, wk, wv : list of list of list of float
        Per-head scalar projections; ``wq[h]`` is ``c x cs`` (line 1).
    wqp, wkp : list of list of list of list of float
        Per-head, per-point query and key point projections; ``wqp[h][p]``
        is ``3 x cs`` (line 2).  ``len(wqp[0])`` is the number of query
        points.
    wvp : list of list of list of list of float
        Per-head, per-point value projections, ``3 x cs`` (line 3).
        ``len(wvp[0])`` is the number of point values.
    wb : list of list of float
        Per-head pair-bias projection, ``nhead x cz`` (line 4).
    gamma : list of float
        Per-head scalar weighting the point term of line 7.  The model
        obtains it through a softplus; the caller passes the value after
        that transform, so nothing here is implicit.
    wo : list of list of float
        Output projection (line 11), with input width
        ``nhead * (cz + c + 4 * npv)``.

    Returns
    -------
    result : RichResult
        Keys: ``s`` (the update ``s~``), ``attn``, ``points`` (the local
        output points of line 10), ``estimate``, ``n``, ``method``.

    Notes
    -----
    The invariance argument needs the attention weights to sum to one, so
    the softmax is not incidental to it.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithm 22
    """
    n = len(s)
    cz = len(z[0][0])
    nh = len(wq)
    c = len(wq[0])
    nqp = len(wqp[0])
    npv = len(wvp[0])
    scale = 1.0 / math.sqrt(c)
    wC = math.sqrt(2.0 / (9.0 * nqp))          # line 5
    wL = math.sqrt(1.0 / 3.0)                  # line 6

    q = [[A.lin(s[i], wq[h]) for i in range(n)] for h in range(nh)]
    k = [[A.lin(s[i], wk[h]) for i in range(n)] for h in range(nh)]
    v = [[A.lin(s[i], wv[h]) for i in range(n)] for h in range(nh)]
    # points are produced in the local frame and then placed globally
    gq = [[[A.rapply(frames[i], A.lin(s[i], wqp[h][p])) for p in range(nqp)]
           for i in range(n)] for h in range(nh)]
    gk = [[[A.rapply(frames[i], A.lin(s[i], wkp[h][p])) for p in range(nqp)]
           for i in range(n)] for h in range(nh)]
    gv = [[[A.rapply(frames[i], A.lin(s[i], wvp[h][p])) for p in range(npv)]
           for i in range(n)] for h in range(nh)]
    b = [[[A.vdot(wb[h], z[i][j]) for j in range(n)] for i in range(n)]
         for h in range(nh)]

    attn, out = [], []
    pts = []
    for h in range(nh):
        ah, ph = [], []
        for i in range(n):
            logits = []
            for j in range(n):
                # line 7: scalar term, pair bias, and the squared point
                # distance measured in the global frame
                dsq = sum(A.vnorm2(A.vsub(gq[h][i][p], gk[h][j][p]))
                          for p in range(nqp))
                logits.append(wL * (scale * A.vdot(q[h][i], k[h][j])
                                    + b[h][i][j]
                                    - 0.5 * gamma[h] * wC * dsq))
            a = A.smax(logits)
            ah.append(a)
            # line 10: average the globally placed value points, then map
            # back through the inverse of the receiving frame
            row = []
            for p in range(npv):
                acc = [sum(a[j] * gv[h][j][p][t] for j in range(n))
                       for t in range(3)]
                row.append(A.rinvapply(frames[i], acc))
            ph.append(row)
        attn.append(ah)
        pts.append(ph)

    for i in range(n):
        cat = []
        for h in range(nh):
            a = attn[h][i]
            # line 8: pair-weighted sum
            cat.extend([sum(a[j] * z[i][j][t] for j in range(n))
                        for t in range(cz)])
            # line 9: value-weighted sum
            cat.extend([sum(a[j] * v[h][j][t] for j in range(n))
                        for t in range(c)])
            # line 11: the local points and their norms
            for p in range(npv):
                cat.extend(pts[h][i][p])
            for p in range(npv):
                cat.append(math.sqrt(A.vnorm2(pts[h][i][p])))
        out.append(A.lin(cat, wo))

    cs = len(out[0])
    flat = [out[i][t] for i in range(n) for t in range(cs)]
    return RichResult(
        payload={
            "s": out,
            "attn": attn,
            "points": pts,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "method": "AlphaFold invariant point attention",
        }
    )


def cheatsheet():
    return "alfipa: invariant point attention over rigid backbone frames"
