# morie.fn -- function file (rootcoder007/morie)
"""All-atom coordinates from backbone frames and torsion angles (AlphaFold)."""

from __future__ import annotations

import math

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_sidechain"]


def _rotx(a):
    """Rotation about the x-axis from a unit 2-vector -- Algorithm 25.

    The network predicts an unnormalised ``(cos, sin)`` pair; it is
    normalised here, which is what makes the result a rotation.
    """
    nrm = math.sqrt(a[0] * a[0] + a[1] * a[1])
    c, s = a[0] / nrm, a[1] / nrm
    return [[[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], [0.0, 0.0, 0.0]]


def alphafold_sidechain(frames, angles, littf, parent, litx, frameof):
    """Compute all atom coordinates -- Algorithms 24 and 25, pp. 30-31.

    Side chain atoms are placed by walking the torsion hierarchy: each
    torsion frame is its parent frame composed with a literature transform
    and a rotation about the torsion axis, which the construction puts on
    the x-axis.  Bond lengths and angles come from the caller-supplied
    literature tables, so only the torsions are free -- that is the whole
    point of the parameterisation.

    This implements the algebra, not the amino-acid tables: the idealised
    geometry is an argument, never baked in.

    Parameters
    ----------
    frames : list
        Backbone frames, one ``[R, t]`` per residue.
    angles : list of list of list of float
        Torsion angles, ``n x nframe x 2``, each an unnormalised
        ``(cos, sin)`` pair as the network emits them.
    littf : list
        Literature transform of each torsion frame into its parent,
        one ``[R, t]`` per frame.
    parent : list of int
        Parent frame index for each torsion frame; ``-1`` means the frame
        hangs directly off the backbone frame.
    litx : list of list of float
        Idealised atom position within its own frame, length 3 each.
    frameof : list of int
        Index of the torsion frame each atom belongs to.

    Returns
    -------
    result : RichResult
        Keys: ``x`` (atom coordinates, ``n x natoms x 3``), ``frames``
        (the composed torsion frames, ``n x nframe``), ``estimate``,
        ``n``, ``method``.

    Notes
    -----
    Two properties anchor this and the parity harness checks both: every
    composed frame is a proper rotation, and applying one global rigid
    motion to all backbone frames moves every atom by exactly that motion,
    which is the equivariance the structure module relies on.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithms 24-25
    """
    n = len(frames)
    nf = len(littf)
    allf, allx = [], []
    for i in range(n):
        tf = [None] * nf
        for f in range(nf):
            base = frames[i] if parent[f] < 0 else tf[parent[f]]
            if base is None:
                raise ValueError("frame %d referenced before its parent" % f)
            tf[f] = A.rcompose(A.rcompose(base, littf[f]), _rotx(angles[i][f]))
        allf.append(tf)
        allx.append([A.rapply(tf[frameof[a]], litx[a])
                     for a in range(len(litx))])

    flat = [allx[i][a][t] for i in range(n) for a in range(len(litx))
            for t in range(3)]
    return RichResult(
        payload={
            "x": allx,
            "frames": allf,
            "estimate": sum(flat) / len(flat),
            "n": n,
            "method": "AlphaFold all-atom coordinates from torsion angles",
        }
    )


def cheatsheet():
    return "alfsdc: side chain and backbone atoms from torsion angles"
