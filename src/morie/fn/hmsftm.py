# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Softmax function normalizes class scores into probabilities."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_softmax_function"]


def geron_softmax_function(scores, axis=-1):
    """
    Softmax function normalizes class scores into probabilities.

    Formula: p_k = exp(s_k) / sum_j exp(s_j)

    Shift-invariant form: the row maximum is subtracted before
    exponentiating, which is exact (softmax is invariant to a constant
    shift) and keeps large scores from overflowing. The Jacobian
    ``diag(p) - p p^T`` is returned for the 1-D case, which is what a
    softmax layer backpropagates.

    Parameters
    ----------
    scores : array-like
        Class scores; softmax is taken along `axis`.
    axis : int, default -1
        Axis holding the classes.

    Returns
    -------
    result : RichResult
        Keys: p, argmax, jacobian, estimate, n, method.

    Examples
    --------
    >>> r = geron_softmax_function([1.0, 2.0, 3.0])
    >>> [round(float(v), 6) for v in r["p"]]
    [0.090031, 0.244728, 0.665241]
    >>> round(float(sum(r["p"])), 12)
    1.0
    >>> [round(float(v), 6) for v in geron_softmax_function([0.0, 0.0])["p"]]
    [0.5, 0.5]
    >>> round(float(geron_softmax_function([1.0, 2.0, 3.0])["jacobian"][2][2]), 6)
    0.222695

    References
    ----------
    Géron Ch 4
    """
    s = np.atleast_1d(np.asarray(scores, dtype=float))
    if s.size == 0:
        raise ValueError("geron_softmax_function: scores is empty")
    if not np.all(np.isfinite(s)):
        raise ValueError("geron_softmax_function: scores contains non-finite values")
    ax = int(axis)
    if not (-s.ndim <= ax < s.ndim):
        raise ValueError(f"geron_softmax_function: axis {ax} is out of range for a {s.ndim}-D array")
    if s.shape[ax] < 2:
        raise ValueError(
            f"geron_softmax_function: softmax needs at least 2 classes along axis {ax}, got {s.shape[ax]}"
        )

    e = np.exp(s - np.max(s, axis=ax, keepdims=True))
    p = e / np.sum(e, axis=ax, keepdims=True)
    jac = np.diag(p) - np.outer(p, p) if s.ndim == 1 else None

    return RichResult(
        title="Softmax",
        summary_lines=[("Classes", int(s.shape[ax])), ("Max probability", float(np.max(p)))],
        interpretation="Softmax outputs are positive and sum to one along the class axis; it is shift-invariant.",
        payload={
            "p": p,
            "probabilities": p,
            "argmax": np.argmax(p, axis=ax),
            "jacobian": jac,
            "estimate": float(np.max(p)),
            "n": int(s.shape[ax]),
            "method": "Softmax with max-shift stabilisation (exact, not an approximation)",
        },
    )


def cheatsheet():
    return "hmsftm: Softmax function normalizes class scores into probabilities"
