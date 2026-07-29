# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.20: the gender-projection regulariser."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_gender_projection_reg"]


def kamath_ch6_gender_projection_reg(W_stereo, g):
    """R = sum_{w in W_stereo} (g/||g||) w^T.

    The stereotypical embeddings' projections onto the unit gender
    direction, summed: driving R to 0 orthogonalises them against the
    axis. Each term is the scalar (g_hat . w).

    NOTE the printed equation writes the summand as "(g/||g||) w^T",
    which is only conformable as an inner product if w is a row vector
    -- as written with column vectors it would be an outer product,
    which cannot be a scalar regulariser. The inner-product reading is
    implemented, and the SIGNED sum is the estimate; because signed
    projections can cancel, ``sum_abs`` and the per-word projections
    are returned alongside.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.20, printed
    p. 243.

    Examples
    --------
    >>> out = kamath_ch6_gender_projection_reg(
    ...     [[1.0, 0.0], [3.0, 0.0]], [2.0, 0.0])
    >>> out["estimate"], out["projections"]
    (4.0, [1.0, 3.0])
    >>> kamath_ch6_gender_projection_reg([[0.0, 5.0]], [2.0, 0.0]
    ...                                  )["estimate"]
    0.0
    """
    gv = np.atleast_1d(np.asarray(g, dtype=float))
    W = np.atleast_2d(np.asarray(W_stereo, dtype=float))
    if W.shape[0] == 0:
        raise ValueError("W_stereo is empty; a sum over no stereotypical "
                         "words is undefined, not 0.")
    norm = float(np.linalg.norm(gv))
    if norm == 0:
        raise ValueError("g is the zero vector; there is no direction to "
                         "project onto.")
    if W.shape[1] != gv.shape[0]:
        raise ValueError(
            f"W_stereo has width {W.shape[1]} but g has {gv.shape[0]}.")
    proj = W @ (gv / norm)
    return RichResult(payload={
        "estimate": float(proj.sum()),
        "projections": [float(v) for v in proj],
        "sum_abs": float(np.abs(proj).sum()),
        "g_norm": norm, "n": int(W.shape[0]),
        "method": "gender-projection regulariser (Kamath Eq 6.20)"})


def cheatsheet():
    return "km096: sum of stereotype embeddings' projections onto g_hat"
