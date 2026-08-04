# morie.fn -- function file (rootcoder007/morie)
"""Triangular multiplicative update of the AlphaFold pair representation."""

from __future__ import annotations

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_triangle_mult"]


def alphafold_triangle_mult(z, wag, wav, wbg, wbv, wg, wo, mode="outgoing",
                            layernorm=True):
    """Triangular multiplicative update -- Algorithms 11 and 12, p. 18.

    Each edge ``ij`` of the pair graph is updated from the other two edges
    of every triangle it takes part in.  The "outgoing" form (Algorithm 11)
    contracts over ``a_ik * b_jk``; the "incoming" form (Algorithm 12)
    contracts over ``a_ki * b_kj``.  These are the only two lines that
    differ between the algorithms.

    Every weight is supplied by the caller, so the result is a fixed
    function of its arguments.

    Parameters
    ----------
    z : list of list of list of float
        Pair representation, ``n x n x cz``.
    wag, wav, wbg, wbv : list of list of float
        Gate and value projections for the left ("a") and right ("b")
        edges, each ``c x cz``.  Line 2 of both algorithms multiplies a
        sigmoid gate elementwise into a value projection.
    wg : list of list of float
        Output gate projection, ``cz x cz`` (line 3).
    wo : list of list of float
        Output projection, ``cz x c`` (line 4).
    mode : {"outgoing", "incoming"}
        Which of Algorithm 11 / Algorithm 12 to apply.
    layernorm : bool
        Apply the layer normalisations of lines 1 and 4.  The spec always
        does; setting it to ``False`` exposes the degenerate reduction
        described below, in which the update is a plain matrix product.

    Returns
    -------
    result : RichResult
        Keys: ``z`` (the update ``z~``, ``n x n x cz``), ``estimate``
        (mean of ``z~``), ``n``, ``method``.

    Notes
    -----
    In the single-channel case ``c = cz = 1`` with ``layernorm=False``,
    zero gate weights, unit value weights and unit output weight, the
    update collapses to ``0.125 * (Z Z')`` for "outgoing" and
    ``0.125 * (Z' Z)`` for "incoming" -- the two sigmoid gates and the
    output gate each contribute a factor of one half.  That closed form is
    used as an anchor in the parity harness.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithms 11-12
    """
    if mode not in ("outgoing", "incoming"):
        raise ValueError("mode must be 'outgoing' or 'incoming'")
    n = len(z)
    cz = len(z[0][0])
    zn = [[A.lnorm(z[i][j]) if layernorm else list(z[i][j])
           for j in range(n)] for i in range(n)]

    # line 2: gated left and right edge projections
    a = [[[A.sigm(x) * y for x, y in zip(A.lin(zn[i][j], wag),
                                         A.lin(zn[i][j], wav))]
          for j in range(n)] for i in range(n)]
    b = [[[A.sigm(x) * y for x, y in zip(A.lin(zn[i][j], wbg),
                                         A.lin(zn[i][j], wbv))]
          for j in range(n)] for i in range(n)]
    # line 3: output gate
    g = [[[A.sigm(x) for x in A.lin(zn[i][j], wg)] for j in range(n)]
         for i in range(n)]

    c = len(a[0][0])
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            # line 4: the triangle contraction, the only differing line
            if mode == "outgoing":
                s = [sum(a[i][k][q] * b[j][k][q] for k in range(n))
                     for q in range(c)]
            else:
                s = [sum(a[k][i][q] * b[k][j][q] for k in range(n))
                     for q in range(c)]
            if layernorm:
                s = A.lnorm(s)
            p = A.lin(s, wo)
            row.append([g[i][j][q] * p[q] for q in range(cz)])
        out.append(row)

    flat = [out[i][j][q] for i in range(n) for j in range(n) for q in range(cz)]
    return RichResult(
        payload={
            "z": out,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "mode": mode,
            "method": "AlphaFold triangular multiplicative update (%s)" % mode,
        }
    )


def cheatsheet():
    return "alftrm: triangular multiplicative update, outgoing and incoming"
