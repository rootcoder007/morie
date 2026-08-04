# morie.fn -- function file (rootcoder007/morie)
"""Frame Aligned Point Error (FAPE), the main AlphaFold structural loss."""

from __future__ import annotations

import math

from . import _alfcore as A
from ._richresult import RichResult

__all__ = ["alphafold_fape_loss"]


def alphafold_fape_loss(frames_pred, x, frames_true, x_true, Z=10.0,
                        dclamp=10.0, eps=1e-4):
    """Frame aligned point error -- Algorithm 28, p. 34.

    Every predicted atom is expressed in the local frame of every predicted
    residue, the same is done for the ground truth, and the two clouds are
    compared pointwise.  Because each atom is scored under every frame, the
    loss is sensitive to global arrangement as well as local geometry, and
    unlike an RMSD it needs no superposition.  It also distinguishes
    mirror images (supplement equations 16-17).

    Parameters
    ----------
    frames_pred, frames_true : list
        Predicted and ground-truth frames, each ``[R, t]``.
    x, x_true : list of list of float
        Predicted and ground-truth atom positions, each length 3.
    Z : float
        Length scale, 10 angstrom in the spec.
    dclamp : float
        Distance clamp, 10 angstrom in the spec.
    eps : float
        Added under the square root of line 3 to keep the gradient finite
        at zero.  Note that it makes the loss ``sqrt(eps) / Z`` rather than
        exactly zero for a perfect prediction; pass ``eps=0`` for the
        unregularised form.

    Returns
    -------
    result : RichResult
        Keys: ``estimate`` (the loss), ``d`` (the ``nframes x natoms``
        distance matrix), ``nframes``, ``natoms``, ``method``.

    Notes
    -----
    Three closed forms anchor this, and the parity harness checks all
    three: the loss is ``sqrt(eps) / Z`` when prediction and truth
    coincide (exactly zero when ``eps = 0``); it equals ``dclamp / Z`` when
    every pair is beyond the clamp; and it is unchanged when one global
    rigid motion is applied to the predicted frames and points together,
    since the local coordinates of line 1 are then unchanged.

    References
    ----------
    Jumper et al (2021) Nature 596:583-589, Supplementary Algorithm 28
    """
    nf = len(frames_pred)
    na = len(x)
    d = []
    tot = 0.0
    for i in range(nf):
        row = []
        for j in range(na):
            xi = A.rinvapply(frames_pred[i], x[j])          # line 1
            xt = A.rinvapply(frames_true[i], x_true[j])     # line 2
            dij = math.sqrt(A.vnorm2(A.vsub(xi, xt)) + eps)  # line 3
            row.append(dij)
            tot += dij if dij < dclamp else dclamp          # line 4
        d.append(row)

    loss = tot / (nf * na) / Z
    return RichResult(
        payload={
            "estimate": loss,
            "d": d,
            "nframes": nf,
            "natoms": na,
            "method": "AlphaFold frame aligned point error",
        }
    )


def cheatsheet():
    return "alffap: frame aligned point error, the clamped AlphaFold loss"
