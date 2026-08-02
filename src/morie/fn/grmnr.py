# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Max-norm regularization."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_max_norm_regularization"]

_METHOD = "Max-norm weight projection"


def geron_max_norm_regularization(W, r, axis=1):
    r"""Project each weight vector onto the L2 ball of radius ``r``.

    .. math::
        \text{if } \|w_i\|_2 > r: \quad w_i \leftarrow w_i\,\frac{r}{\|w_i\|_2}

    This is a *constraint*, not a penalty: it is applied after the
    gradient step and adds nothing to the loss.  So unlike weight decay
    it never shrinks a weight vector that is already inside the ball,
    and it cannot be traded off against the data term.

    The projection is per *unit* -- one row of ``W`` in the
    ``(out, in)`` layout used by :mod:`morie.fn.grlinf` -- so a single
    over-eager unit is reined in without touching its neighbours.  The
    same rescale-don't-truncate logic as gradient clipping
    (:mod:`morie.fn.grgcl`), applied to weights instead of gradients.

    Parameters
    ----------
    W : array-like, shape (out, in)
    r : float
        Positive radius.
    axis : int, optional
        Axis to take the norm over. Default 1 (per output unit).

    Returns
    -------
    RichResult
        Payload keys ``W_new``, ``norms_before``, ``norms_after``,
        ``n_projected``, ``rows_projected``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 11, Max-Norm Regularization section.

    Examples
    --------
    The first unit is over the limit and is rescaled to exactly ``r``;
    the second is inside and is left alone:

    >>> r_ = geron_max_norm_regularization([[3.0, 4.0], [0.1, 0.0]], r=1.0)
    >>> [[round(v, 10) for v in row] for row in r_["W_new"]]
    [[0.6, 0.8], [0.1, 0.0]]
    >>> [round(v, 10) for v in r_["norms_after"]]
    [1.0, 0.1]
    >>> r_["rows_projected"]
    [0]

    A larger radius leaves everything untouched -- the constraint is
    inactive:

    >>> geron_max_norm_regularization([[3.0, 4.0]], r=10.0)["n_projected"]
    0
    """
    A = np.atleast_2d(np.asarray(W, dtype=float))
    if A.ndim != 2:
        raise ValueError(f"W must be 2-D of shape (out, in), got shape {A.shape}.")
    if A.size == 0:
        raise ValueError("W is empty.")
    if not np.all(np.isfinite(A)):
        raise ValueError("W must be finite.")
    r = float(r)
    if not np.isfinite(r) or r <= 0:
        raise ValueError(f"r must be a positive finite radius, got {r}.")
    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}.")

    norms = np.linalg.norm(A, axis=axis, keepdims=True)
    scale = np.where(norms > r, r / np.where(norms > 0, norms, 1.0), 1.0)
    B = A * scale
    after = np.linalg.norm(B, axis=axis)
    hit = np.flatnonzero((norms > r).ravel())
    if np.any(after > r * (1.0 + 1e-9)):
        raise ValueError("a projected row still exceeds r; this is a bug.")

    return RichResult(
        title="Max-norm regularization",
        summary_lines=[("Radius", r), ("Rows projected", int(hit.size))],
        payload={
            "W_new": B.tolist(),
            "norms_before": norms.ravel().tolist(),
            "norms_after": after.tolist(),
            "n_projected": int(hit.size),
            "rows_projected": hit.tolist(),
            "r": r,
            "estimate": B.tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmnr: per-row w *= min(1, r/||w||); a constraint applied after the step, not a penalty"
