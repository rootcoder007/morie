# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient clipping by global L2 norm."""

from . import _array_core as np

from ._richresult import RichResult
from .grdcl import gradient_clipping

__all__ = ["geron_gradient_clipping"]

_METHOD = "Global-norm gradient clipping"


def geron_gradient_clipping(gradients, c):
    r"""Rescale the gradient if -- and only if -- it is too long.

    .. math::
        \text{if } \|g\|_2 > c: \quad g \leftarrow g\,\frac{c}{\|g\|_2}

    Two properties matter and both are checked here.  The clipped norm
    is at most ``c``; and the *direction* is untouched, because every
    component is multiplied by the same scalar.  Clipping each component
    separately (``clip_by_value``) does not have the second property --
    it bends the update away from the true gradient -- which is why the
    global-norm version is the default for recurrent nets.

    The arithmetic is delegated to
    :func:`morie.fn.grdcl.gradient_clipping`; this wrapper adds the
    Géron-facing signature and the direction check.

    Parameters
    ----------
    gradients : array-like, or sequence of array-like
        A single tensor, or a list of tensors treated jointly as one
        global vector (the usual multi-layer case).
    c : float
        Positive clipping threshold.

    Returns
    -------
    RichResult
        Payload keys ``clipped``, ``total_norm``, ``clipped_norm``,
        ``clip_coef``, ``was_clipped``, ``cosine_with_original``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 11, Gradient Clipping section (Pascanu et al. 2013).

    Examples
    --------
    A 3-4-5 gradient clipped to 1 keeps its direction exactly:

    >>> r = geron_gradient_clipping([3.0, 4.0], c=1.0)
    >>> r["total_norm"]
    5.0
    >>> [round(v, 10) for v in r["clipped"]]
    [0.6, 0.8]
    >>> round(r["clipped_norm"], 10), round(r["cosine_with_original"], 10)
    (1.0, 1.0)

    Under the threshold, nothing happens -- clipping is a ceiling, not a
    normalisation:

    >>> r2 = geron_gradient_clipping([0.3, 0.4], c=1.0)
    >>> r2["was_clipped"], r2["clipped"]
    (False, [0.3, 0.4])
    """
    c = float(c)
    if not np.isfinite(c) or c <= 0:
        raise ValueError(f"c must be a positive finite threshold, got {c}.")
    if isinstance(gradients, (list, tuple)) and len(gradients) == 0:
        raise ValueError("gradients is empty.")
    listed = isinstance(gradients, (list, tuple)) and any(
        np.asarray(g).ndim > 0 for g in gradients
    )
    flat = np.concatenate([np.asarray(g, dtype=float).ravel() for g in gradients]) \
        if listed else np.asarray(gradients, dtype=float).ravel()
    if flat.size == 0:
        raise ValueError("gradients is empty.")
    if not np.all(np.isfinite(flat)):
        raise ValueError(
            "gradients contain non-finite values; clipping cannot rescue an "
            "inf/nan gradient, fix the forward pass."
        )

    inner = gradient_clipping(gradients, max_norm=c)
    clipped = inner["tensor"]
    flat_c = np.concatenate([np.asarray(g, dtype=float).ravel() for g in clipped]) \
        if listed else np.asarray(clipped, dtype=float).ravel()

    n0 = float(np.linalg.norm(flat))
    n1 = float(np.linalg.norm(flat_c))
    cos = float(flat @ flat_c / (n0 * n1)) if n0 > 0 and n1 > 0 else 1.0
    if n1 > c * (1.0 + 1e-9):
        raise ValueError(f"clipped norm {n1} exceeds the threshold {c}; this is a bug.")

    return RichResult(
        title="Gradient clipping (global norm)",
        summary_lines=[("||g||", n0), ("Threshold", c), ("Clipped", n0 > c)],
        payload={
            "clipped": [np.asarray(g, dtype=float).tolist() for g in clipped]
            if listed else np.asarray(clipped, dtype=float).tolist(),
            "total_norm": n0,
            "clipped_norm": n1,
            "clip_coef": float(inner["clip_coef"]),
            "was_clipped": bool(n0 > c),
            "cosine_with_original": cos,
            "threshold": c,
            "estimate": n1,
            "n": int(flat.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grgcl: g *= min(1, c/||g||) -- norm capped, direction preserved (delegates to grdcl)"
