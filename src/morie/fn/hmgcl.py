# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient clipping by global norm to stabilize training."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_gradient_clipping"]


def geron_gradient_clipping(grads, max_norm, norm_type=2.0):
    """
    Gradient clipping by global norm to stabilize training.

    Formula: if ||g|| > c: g <- g * c / ||g||

    The norm is *global*: every parameter array is treated as one long
    vector, so clipping rescales all of them by the same factor and the
    gradient direction is preserved exactly. Per-array clipping would
    change the direction, which is why this variant is the one that keeps
    a step interpretable.

    Parameters
    ----------
    grads : array-like or sequence of array-like
        Gradients; a ragged sequence of arrays is accepted and returned
        with the same structure.
    max_norm : float
        Clipping threshold ``c``; must be positive and finite.
    norm_type : float, default 2.0
        Order of the norm; ``inf`` gives max-abs clipping.

    Returns
    -------
    result : RichResult
        Keys: clipped, total_norm, scale, was_clipped, new_norm,
        estimate, n, method.

    Examples
    --------
    >>> r = geron_gradient_clipping([3.0, 4.0], max_norm=1.0)
    >>> round(float(r["total_norm"]), 12)
    5.0
    >>> [round(float(v), 12) for v in r["clipped"]]
    [0.6, 0.8]
    >>> r["was_clipped"]
    True
    >>> geron_gradient_clipping([0.3, 0.4], max_norm=1.0)["was_clipped"]
    False
    >>> [round(float(v), 12) for v in geron_gradient_clipping([0.3, 0.4], 1.0)["clipped"]]
    [0.3, 0.4]

    References
    ----------
    Géron Ch 11
    """
    c = float(max_norm)
    if not np.isfinite(c) or c <= 0:
        raise ValueError(f"geron_gradient_clipping: max_norm must be positive and finite, got {max_norm!r}")
    p = float(norm_type)
    if p <= 0:
        raise ValueError(f"geron_gradient_clipping: norm_type must be positive, got {norm_type!r}")

    single = False
    if isinstance(grads, np.ndarray) or not isinstance(grads, (list, tuple)):
        arrays = [np.atleast_1d(np.asarray(grads, dtype=float))]
        single = True
    elif len(grads) and all(np.ndim(g) == 0 for g in grads):
        arrays = [np.asarray(grads, dtype=float)]
        single = True
    else:
        arrays = [np.atleast_1d(np.asarray(g, dtype=float)) for g in grads]
    if not arrays or all(a.size == 0 for a in arrays):
        raise ValueError("geron_gradient_clipping: grads is empty")
    flat = np.concatenate([a.ravel() for a in arrays])
    if not np.all(np.isfinite(flat)):
        raise ValueError("geron_gradient_clipping: grads contains non-finite values")

    if np.isinf(p):
        total = float(np.max(np.abs(flat)))
    else:
        total = float(np.sum(np.abs(flat) ** p) ** (1.0 / p))
    scale = 1.0 if total <= c else c / total
    clipped = [a * scale for a in arrays]
    out = clipped[0] if single else clipped

    return RichResult(
        title="Gradient clipping",
        summary_lines=[("Global norm", total), ("Threshold", c), ("Scale", scale)],
        interpretation="Rescaling is uniform, so the update direction is unchanged.",
        payload={
            "clipped": out,
            "total_norm": total,
            "scale": float(scale),
            "was_clipped": bool(scale < 1.0),
            "new_norm": float(total * scale),
            "max_norm": c,
            "norm_type": p,
            "estimate": float(total * scale),
            "n": int(flat.size),
            "method": "global-norm gradient clipping",
        },
    )


def cheatsheet():
    return "hmgcl: Gradient clipping by global norm to stabilize training"
