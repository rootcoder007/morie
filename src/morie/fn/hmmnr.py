# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Max-norm regularization: rescale weights so ||w||_2 <= r."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_max_norm"]

_METHOD = "Max-norm regularization (projection onto the l2 ball)"


def geron_max_norm(w, r, axis=None):
    """
    Max-norm regularization: rescale weights so ||w||_2 <= r.

    Formula: if ||w||>r: w <- w * r / ||w||

    A hard constraint applied after each update rather than a penalty
    added to the cost: the incoming weight vector of every unit is
    projected back onto the l2 ball of radius ``r``.  Because it is a
    projection, the *direction* of the weight vector is untouched --
    only its length changes -- which is what separates it from L2 decay,
    where every step shrinks the vector whether it needed shrinking or
    not.

    With ``axis=0`` each column (one unit's fan-in) is constrained
    separately, which is the usual Keras ``MaxNorm`` behaviour.

    Parameters
    ----------
    w : array-like
        Weights.
    r : float
        Maximum allowed l2 norm (positive).
    axis : int, optional
        Axis to reduce over when computing norms.  ``None`` treats the
        whole array as one vector.

    Returns
    -------
    result : RichResult
        Keys: w, norm_before, norm_after, clipped, n_clipped,
        estimate, n, method.

    Examples
    --------
    ``[3, 4]`` has norm 5; clipping to 2 scales by 2/5:

    >>> r = geron_max_norm([3.0, 4.0], r=2.0)
    >>> [round(float(v), 9) for v in r["w"]]
    [1.2, 1.6]
    >>> float(r["norm_before"]), float(r["norm_after"])
    (5.0, 2.0)
    >>> r["clipped"]
    True

    A vector already inside the ball is returned untouched:

    >>> u = geron_max_norm([0.3, 0.4], r=1.0)
    >>> [float(v) for v in u["w"]], u["clipped"]
    ([0.3, 0.4], False)

    Per-column constraint: the first column (norm 5) is clipped, the
    second (norm 1) is not:

    >>> c = geron_max_norm([[3.0, 1.0], [4.0, 0.0]], r=1.0, axis=0)
    >>> [[round(float(v), 6) for v in row] for row in c["w"]]
    [[0.6, 1.0], [0.8, 0.0]]
    >>> c["n_clipped"]
    1

    References
    ----------
    Géron Ch 11
    """
    W = np.atleast_1d(np.asarray(w, dtype=float))
    if W.size == 0:
        raise ValueError("geron_max_norm: w is empty")
    if not np.all(np.isfinite(W)):
        raise ValueError("geron_max_norm: w contains non-finite values")
    radius = float(r)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"geron_max_norm: r must be a positive finite radius, got {r!r}")

    if axis is None:
        norm_before = float(np.linalg.norm(W))
        if norm_before > radius:
            out = W * (radius / norm_before)
            clipped = True
            n_clipped = 1
        else:
            out = W.copy()
            clipped = False
            n_clipped = 0
        norm_after = float(np.linalg.norm(out))
    else:
        ax = int(axis)
        if not (-W.ndim <= ax < W.ndim):
            raise ValueError(f"geron_max_norm: axis {axis} is out of range for an array with ndim={W.ndim}")
        norms = np.linalg.norm(W, axis=ax, keepdims=True)
        factor = np.where(norms > radius, radius / np.where(norms == 0, 1.0, norms), 1.0)
        out = W * factor
        norm_before = np.squeeze(norms, axis=ax)
        norm_after = np.linalg.norm(out, axis=ax)
        n_clipped = int(np.count_nonzero(np.squeeze(norms, axis=ax) > radius))
        clipped = n_clipped > 0

    return RichResult(
        title="Max-norm regularization",
        summary_lines=[("Radius r", radius), ("Vectors clipped", n_clipped)],
        interpretation="A projection, not a penalty: the direction of each weight vector is preserved exactly.",
        payload={
            "w": out,
            "norm_before": norm_before,
            "norm_after": norm_after,
            "clipped": clipped,
            "n_clipped": n_clipped,
            "r": radius,
            "estimate": float(np.max(norm_after)),
            "n": int(W.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmnr: max-norm projection w <- w*r/||w|| when ||w|| > r (direction preserved)"
