# morie.fn -- function file (rootcoder007/morie)
"""Transition layer of the AlphaFold structure module."""

from __future__ import annotations

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_structure_transition"]


def alphafold_structure_transition(s, w1, w2, w3, layernorm=True, drop=None):
    """Structure module transition -- Algorithm 20 lines 7-9, p. 26.

    A three-layer residual MLP applied to the single representation between
    invariant point attention and the backbone update, followed by layer
    normalisation.  Unlike the two-layer transitions of the Evoformer this
    one keeps the channel width constant throughout.

    All weights are supplied by the caller.  Dropout is applied only if the
    caller passes an explicit mask, so inference is deterministic.

    Parameters
    ----------
    s : list of list of float
        Single representation, ``n x cs``.
    w1, w2, w3 : list of list of float
        The three projections of line 8, each ``cs x cs``.
    layernorm : bool
        Apply the layer normalisation of line 9.
    drop : list of list of float, optional
        Multiplicative dropout mask applied before the normalisation.

    Returns
    -------
    result : RichResult
        Keys: ``s``, ``estimate``, ``n``, ``method``.

    Notes
    -----
    With ``w3`` zero the residual vanishes and the layer reduces to the
    layer normalisation alone, or to the identity when that is off too.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithm 20
    """
    n = len(s)
    out = []
    for i in range(n):
        h = A.lin(s[i], w1)
        h = A.lin([A.relu(t) for t in h], w2)
        h = A.lin([A.relu(t) for t in h], w3)
        u = A.vadd(s[i], h)
        if drop is not None:
            u = [u[t] * drop[i][t] for t in range(len(u))]
        out.append(A.lnorm(u) if layernorm else u)

    cs = len(out[0])
    flat = [out[i][t] for i in range(n) for t in range(cs)]
    return RichResult(
        payload={
            "s": out,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "method": "AlphaFold structure module transition",
        }
    )


def cheatsheet():
    return "alfsts: structure module residual transition and normalisation"
