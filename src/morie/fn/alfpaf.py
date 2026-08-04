# morie.fn -- function file (rootcoder007/morie)
"""Outer product mean: the MSA-to-pair communication step of AlphaFold."""

from __future__ import annotations

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_pair_repr"]


def alphafold_pair_repr(m, wa, wb, wo, layernorm=True):
    """Outer product mean -- Algorithm 10, p. 17.

    This is the only path by which the MSA stack writes into the pair
    stack.  Rows ``i`` and ``j`` of the MSA representation are projected,
    their outer product is averaged over sequences, flattened and projected
    to the pair channel width.

    Every weight is supplied by the caller.

    Parameters
    ----------
    m : list of list of list of float
        MSA representation, ``s x n x cm``.
    wa, wb : list of list of float
        The two projections of line 2, each ``c x cm``.
    wo : list of list of float
        Output projection, ``cz x (c * c)`` (line 4).
    layernorm : bool
        Apply the layer normalisation of line 1.  Setting it to ``False``
        exposes the degenerate reduction described below.

    Returns
    -------
    result : RichResult
        Keys: ``z`` (the pair update, ``n x n x cz``), ``o`` (the flattened
        outer product means, ``n x n x c*c``), ``estimate``, ``n``,
        ``method``.

    Notes
    -----
    With ``c = 1``, ``layernorm=False`` and unit projections, ``o_ij``
    reduces to ``mean_s(m_si m_sj)``, i.e. the Gram matrix ``M' M / s`` --
    a plain matrix product, which the parity harness checks against an
    independently computed product.  The tensor is also symmetric under
    ``i <-> j`` whenever ``wa == wb``.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithm 10
    """
    s = len(m)
    n = len(m[0])
    mn = [[A.lnorm(m[si][i]) if layernorm else list(m[si][i])
           for i in range(n)] for si in range(s)]
    # line 2
    av = [[A.lin(mn[si][i], wa) for i in range(n)] for si in range(s)]
    bv = [[A.lin(mn[si][i], wb) for i in range(n)] for si in range(s)]
    c = len(av[0][0])

    # line 3: outer product, averaged over sequences, then flattened
    o = []
    for i in range(n):
        row = []
        for j in range(n):
            f = []
            for p in range(c):
                for q in range(c):
                    f.append(sum(av[si][i][p] * bv[si][j][q]
                                 for si in range(s)) / s)
            row.append(f)
        o.append(row)

    # line 4
    z = [[A.lin(o[i][j], wo) for j in range(n)] for i in range(n)]
    cz = len(z[0][0])
    flat = [z[i][j][t] for i in range(n) for j in range(n) for t in range(cz)]
    return RichResult(
        payload={
            "z": z,
            "o": o,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "method": "AlphaFold outer product mean (pair representation update)",
        }
    )


def cheatsheet():
    return "alfpaf: outer product mean, MSA to pair communication"
