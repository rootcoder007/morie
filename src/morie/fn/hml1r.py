# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""L1 regularization adds |theta| penalty to cost."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_l1_regularization"]

_METHOD = "L1 (Lasso) regularization penalty"


def geron_l1_regularization(theta, alpha, skip_bias=False):
    """
    L1 regularization adds |theta| penalty to cost.

    Formula: J = J_data + alpha * sum |theta_i|

    Returns the penalty, its subgradient ``alpha * sign(theta)`` -- with
    0 chosen at the kink, the only subgradient value that leaves an
    exactly-zero weight at rest -- and the soft-thresholding proximal
    operator ``sign(theta) * max(|theta| - alpha, 0)``.  The proximal
    form is where the sparsity actually comes from: subgradient descent
    only approaches zero, soft-thresholding lands on it.

    Parameters
    ----------
    theta : array-like
        Parameter vector.
    alpha : float
        Regularization strength (non-negative).
    skip_bias : bool
        If True, ``theta[0]`` is treated as the bias and excluded from
        the penalty, which is the usual convention.

    Returns
    -------
    result : RichResult
        Keys: penalty, gradient, prox, n_zero, estimate, n, method.

    Examples
    --------
    ``alpha * (|3| + |-1| + |0|) = 0.5 * 4 = 2``:

    >>> r = geron_l1_regularization([3.0, -1.0, 0.0], alpha=0.5)
    >>> float(r["penalty"])
    2.0
    >>> [float(v) for v in r["gradient"]]
    [0.5, -0.5, 0.0]

    Soft-thresholding at alpha=0.5 shrinks each weight toward zero by
    0.5 and clips at zero:

    >>> [float(v) for v in r["prox"]]
    [2.5, -0.5, 0.0]
    >>> [abs(float(v)) for v in geron_l1_regularization([0.3, -0.4], alpha=0.5)["prox"]]
    [0.0, 0.0]

    Excluding the bias leaves it out of both penalty and gradient:

    >>> b = geron_l1_regularization([10.0, -1.0], alpha=1.0, skip_bias=True)
    >>> float(b["penalty"]), [float(v) for v in b["gradient"]]
    (1.0, [0.0, -1.0])

    References
    ----------
    Géron Ch 11
    """
    t = np.atleast_1d(np.asarray(theta, dtype=float))
    if t.size == 0:
        raise ValueError("geron_l1_regularization: theta is empty")
    if not np.all(np.isfinite(t)):
        raise ValueError("geron_l1_regularization: theta contains non-finite values")
    a = float(alpha)
    if not np.isfinite(a) or a < 0:
        raise ValueError(f"geron_l1_regularization: alpha must be finite and non-negative, got {alpha!r}")

    mask = np.ones_like(t)
    if skip_bias:
        if t.size < 2:
            raise ValueError("geron_l1_regularization: skip_bias needs at least one non-bias parameter")
        mask.flat[0] = 0.0

    penalty = a * float(np.sum(np.abs(t) * mask))
    grad = a * np.sign(t) * mask
    prox = np.where(mask > 0, np.sign(t) * np.maximum(np.abs(t) - a, 0.0), t)

    return RichResult(
        title="L1 regularization",
        summary_lines=[("alpha", a), ("Penalty", penalty), ("Zeros after prox", int(np.sum(prox == 0)))],
        interpretation="L1 drives weights exactly to zero, so it doubles as feature selection.",
        payload={
            "penalty": penalty,
            "gradient": grad,
            "prox": prox,
            "n_zero": int(np.sum(prox == 0)),
            "estimate": penalty,
            "n": int(t.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hml1r: L1 penalty alpha*sum|theta|, subgradient alpha*sign(theta), soft-threshold prox"
