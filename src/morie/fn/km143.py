# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.15: the frame order modelling (FOM) loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_fom_loss"]


def kamath_ch9_fom_loss(r_i, t_i, R=None, P=None):
    r"""L_FOM = -E[ sum_{i=1..R} log P[r_i, t_i] ].

    ``P`` is the model's timestamp distribution for each reordered
    frame (rows = frames, columns = candidate timestamps); ``r_i`` are
    the reordered-frame indices (rows of P) and ``t_i`` their
    ground-truth timestamps (columns). ``R``, if given, must equal the
    number of reordered frames.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.15, printed
    p. 390; Li et al. (2020b).

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_fom_loss([0, 1], [0, 1],
    ...                           P=[[0.5, 0.5], [0.25, 0.75]])
    >>> abs(out["estimate"] - (math.log(2) - math.log(0.75))) < 1e-12
    True
    """
    if P is None:
        raise ValueError("P= (the frame-by-timestamp probability "
                         "matrix) is required.")
    Pm = np.atleast_2d(np.asarray(P, dtype=float))
    rows = np.atleast_1d(np.asarray(r_i)).astype(int)
    cols = np.atleast_1d(np.asarray(t_i)).astype(int)
    if rows.size == 0:
        raise ValueError("no frames were reordered; the FOM loss is "
                         "over an empty set.")
    if rows.shape != cols.shape:
        raise ValueError(
            f"{rows.size} frame indices but {cols.size} timestamps.")
    if np.any((rows < 0) | (rows >= Pm.shape[0])):
        raise ValueError("a frame index lies outside P.")
    if np.any((cols < 0) | (cols >= Pm.shape[1])):
        raise ValueError("a timestamp index lies outside P.")
    if np.any((Pm < 0) | (Pm > 1)):
        raise ValueError("P holds probabilities and must lie in [0, 1].")
    if R is not None and int(R) != rows.size:
        raise ValueError(
            f"R = {R} contradicts the {rows.size} reordered frames.")
    with np.errstate(divide="ignore"):
        per = -np.log(Pm[rows, cols])
    return RichResult(payload={
        "estimate": float(per.sum()),
        "per_frame": [float(v) for v in per],
        "n_reordered": int(rows.size), "n": int(rows.size),
        "method": "frame order modelling loss (Kamath Eq 9.15)"})


def cheatsheet():
    return "km143: -sum log P[frame, true timestamp] over reordered frames"
