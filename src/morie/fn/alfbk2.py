# morie.fn -- function file (rootcoder007/morie)
"""Backbone frame update from a predicted quaternion (AlphaFold)."""

from __future__ import annotations

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_backbone"]


def alphafold_backbone(s, w, b=None, frames=None):
    """Backbone update -- Algorithm 23, p. 29.

    Six numbers are read off each residue's single representation: three
    quaternion components and a translation.  The leading quaternion
    component is fixed to 1 before normalisation, which guarantees a valid
    unit quaternion without a constraint and biases the layer towards small
    rotations, since zero input gives the identity.

    All weights are supplied by the caller.

    Parameters
    ----------
    s : list of list of float
        Single representation, ``n x cs``.
    w : list of list of float
        Projection to ``(b, c, d, t1, t2, t3)``, so ``6 x cs`` (line 1).
    b : list of float, optional
        Bias for that projection.
    frames : list, optional
        Existing frames to compose with, one ``[R, t]`` per residue.  When
        given, the result is ``T_i o BackboneUpdate(s_i)``, which is how
        line 10 of Algorithm 20 applies the update.  When omitted the bare
        update is returned.

    Returns
    -------
    result : RichResult
        Keys: ``frames`` (list of ``[R, t]``), ``quat`` (the normalised
        quaternions), ``estimate`` (mean translation component), ``n``,
        ``method``.

    Notes
    -----
    Every rotation returned is orthogonal with determinant ``+1``; the
    parity harness checks ``R R' = I`` and ``det R = 1`` directly rather
    than trusting agreement between the two arms.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithm 23
    """
    n = len(s)
    out, quats = [], []
    for i in range(n):
        p = A.lin(s[i], w, b)
        R = A.quat2rot(p[0], p[1], p[2])
        t = [p[3], p[4], p[5]]
        nq = (1.0 + p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) ** 0.5
        quats.append([1.0 / nq, p[0] / nq, p[1] / nq, p[2] / nq])
        T = [R, t]
        if frames is not None:
            T = A.rcompose(frames[i], T)
        out.append(T)

    flat = [out[i][1][t] for i in range(n) for t in range(3)]
    return RichResult(
        payload={
            "frames": out,
            "quat": quats,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "method": "AlphaFold backbone update (quaternion to rigid frame)",
        }
    )


def cheatsheet():
    return "alfbk2: backbone frame update from a predicted quaternion"
