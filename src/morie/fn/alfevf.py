# morie.fn -- function file (rootcoder007/morie)
"""Evoformer block stack (AlphaFold trunk)."""

from __future__ import annotations

from . import _alfcore as A
from .alfpaf import alphafold_pair_repr
from .alfsmd import alphafold_msa_attention
from .alftrm import alphafold_triangle_mult
from .tritta import alphafold_triangle_attn
from ._richresult import RichResult

__all__ = ["alphafold_evoformer"]


def _transition(x, w1, w2):
    """Two-layer transition MLP -- Algorithms 9 and 15.

    LayerNorm, expand, relu, project back.  The expansion factor lives in
    the shape of ``w1``; the spec uses four.
    """
    return A.lin([A.relu(t) for t in A.lin(A.lnorm(x), w1)], w2)


def alphafold_evoformer(m, z, w, nblock=1, drop=None):
    """Evoformer stack -- Algorithm 6, p. 14.

    One block runs, in order: MSA row attention with pair bias, MSA column
    attention, MSA transition, outer product mean into the pair stack, the
    two triangular multiplicative updates, the two triangular attentions,
    and the pair transition.  Every sublayer is a residual update.

    All weights are supplied by the caller and shared across blocks, as in
    the published model.  Dropout is not applied unless the caller passes
    explicit masks, so inference is deterministic.

    Parameters
    ----------
    m : list of list of list of float
        MSA representation, ``s x n x cm``.
    z : list of list of list of float
        Pair representation, ``n x n x cz``.
    w : dict
        Weight bundle.  Keys ``rowq, rowk, rowv, rowg, rowo, rowb`` for
        Algorithm 7; ``colq, colk, colv, colg, colo`` for Algorithm 8;
        ``mt1, mt2`` for the MSA transition; ``opa, opb, opo`` for
        Algorithm 10; ``tmoag, tmoav, tmobg, tmobv, tmog, tmoo`` and the
        ``tmi*`` equivalents for Algorithms 11 and 12; ``tasq, task, tasv,
        tasb, tasg, taso`` and the ``tae*`` equivalents for Algorithms 13
        and 14; ``pt1, pt2`` for Algorithm 15; ``sout`` for line 12.
    nblock : int
        Number of blocks.  Fixed count, no tolerance-based early exit.
    drop : dict, optional
        Multiplicative dropout masks keyed by sublayer name.  Omitted means
        no dropout, which is the inference behaviour.

    Returns
    -------
    result : RichResult
        Keys: ``m``, ``z``, ``s`` (the single representation of line 12),
        ``estimate``, ``method``.

    Notes
    -----
    With every output projection set to zero the stack is exactly the
    identity on ``(m, z)`` for any number of blocks, because each sublayer
    contributes a zero residual.  The parity harness uses that as an anchor
    independent of the implementation.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithm 6
    """
    s = len(m)
    n = len(m[0])
    cm = len(m[0][0])
    cz = len(z[0][0])
    m = [[list(m[si][i]) for i in range(n)] for si in range(s)]
    z = [[list(z[i][j]) for j in range(n)] for i in range(n)]
    dm = drop or {}

    def _dropm(name, upd):
        d = dm.get(name)
        if d is None:
            return upd
        return [[[upd[a][b2][c] * d[a][b2][c] for c in range(len(upd[a][b2]))]
                 for b2 in range(len(upd[a]))] for a in range(len(upd))]

    def _addm(base, upd):
        return [[[base[a][b2][c] + upd[a][b2][c]
                  for c in range(len(base[a][b2]))]
                 for b2 in range(len(base[a]))] for a in range(len(base))]

    for _ in range(nblock):
        # lines 2-4: MSA stack
        u = alphafold_msa_attention(m, w["rowq"], w["rowk"], w["rowv"],
                                    w["rowg"], w["rowo"], z=z, wb=w["rowb"],
                                    mode="row")["m"]
        m = _addm(m, _dropm("row", u))
        u = alphafold_msa_attention(m, w["colq"], w["colk"], w["colv"],
                                    w["colg"], w["colo"], mode="column")["m"]
        m = _addm(m, u)
        u = [[_transition(m[si][i], w["mt1"], w["mt2"]) for i in range(n)]
             for si in range(s)]
        m = _addm(m, u)
        # line 5: communication
        u = alphafold_pair_repr(m, w["opa"], w["opb"], w["opo"])["z"]
        z = _addm(z, u)
        # lines 6-9: pair stack
        u = alphafold_triangle_mult(z, w["tmoag"], w["tmoav"], w["tmobg"],
                                    w["tmobv"], w["tmog"], w["tmoo"],
                                    mode="outgoing")["z"]
        z = _addm(z, _dropm("trimulout", u))
        u = alphafold_triangle_mult(z, w["tmiag"], w["tmiav"], w["tmibg"],
                                    w["tmibv"], w["tmig"], w["tmio"],
                                    mode="incoming")["z"]
        z = _addm(z, _dropm("trimulin", u))
        u = alphafold_triangle_attn(z, w["tasq"], w["task"], w["tasv"],
                                    w["tasb"], w["tasg"], w["taso"],
                                    mode="starting")["z"]
        z = _addm(z, _dropm("triattnstart", u))
        u = alphafold_triangle_attn(z, w["taeq"], w["taek"], w["taev"],
                                    w["taeb"], w["taeg"], w["taeo"],
                                    mode="ending")["z"]
        z = _addm(z, _dropm("triattnend", u))
        # line 10: pair transition
        u = [[_transition(z[i][j], w["pt1"], w["pt2"]) for j in range(n)]
             for i in range(n)]
        z = _addm(z, u)

    # line 12: the single representation is a projection of the first MSA row
    srep = [A.lin(m[0][i], w["sout"]) for i in range(n)]

    fm = [m[a][b2][c] for a in range(s) for b2 in range(n) for c in range(cm)]
    fz = [z[a][b2][c] for a in range(n) for b2 in range(n) for c in range(cz)]
    return RichResult(
        payload={
            "m": m,
            "z": z,
            "s": srep,
            "estimate": (sum(fm) + sum(fz)) / (len(fm) + len(fz)),
            "nblock": nblock,
            "method": "AlphaFold Evoformer stack",
        }
    )


def cheatsheet():
    return "alfevf: Evoformer block stack, MSA and pair towers"
