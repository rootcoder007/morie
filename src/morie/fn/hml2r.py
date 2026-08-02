# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""L2 regularization adds theta^2 penalty."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_l2_regularization"]

_METHOD = "L2 (ridge) regularization penalty"


def geron_l2_regularization(theta, alpha, skip_bias=False, eta=None):
    """
    L2 regularization adds theta^2 penalty.

    Formula: J = J_data + (alpha/2) * sum theta_i^2

    With the ``alpha/2`` convention the gradient is exactly
    ``alpha * theta``, and a plain gradient step becomes
    ``theta <- (1 - eta*alpha) * theta - eta * grad_data``: the
    multiplicative shrink factor is returned for a supplied learning
    rate, since that is the quantity that has to stay in ``(0, 1)`` for
    the update to be stable.  Unlike L1 this never reaches zero exactly,
    only geometrically close to it.

    Parameters
    ----------
    theta : array-like
        Parameter vector.
    alpha : float
        Regularization strength (non-negative).
    skip_bias : bool
        If True, exclude ``theta[0]`` from the penalty.
    eta : float, optional
        Learning rate; if given, ``shrink_factor = 1 - eta*alpha`` is
        computed and a value outside ``(0, 1)`` is reported as a warning.

    Returns
    -------
    result : RichResult
        Keys: penalty, gradient, l2_norm, shrink_factor, estimate, n, method.

    Examples
    --------
    ``(alpha/2) * (9 + 1 + 0) = 0.5/2 * 10 = 2.5``:

    >>> r = geron_l2_regularization([3.0, -1.0, 0.0], alpha=0.5)
    >>> float(r["penalty"])
    2.5
    >>> [float(v) for v in r["gradient"]]
    [1.5, -0.5, 0.0]

    The gradient is the derivative of the penalty, confirmed against a
    central difference at theta[0]:

    >>> h = 1e-6
    >>> up = geron_l2_regularization([3.0 + h, -1.0, 0.0], alpha=0.5)["penalty"]
    >>> dn = geron_l2_regularization([3.0 - h, -1.0, 0.0], alpha=0.5)["penalty"]
    >>> round((up - dn) / (2 * h), 6)
    1.5

    Skipping the bias leaves the first entry untouched:

    >>> b = geron_l2_regularization([10.0, 2.0], alpha=1.0, skip_bias=True)
    >>> float(b["penalty"]), [float(v) for v in b["gradient"]]
    (2.0, [0.0, 2.0])

    The weight-decay view: ``1 - 0.1*0.5 = 0.95`` per step.

    >>> geron_l2_regularization([1.0], alpha=0.5, eta=0.1)["shrink_factor"]
    0.95

    References
    ----------
    Géron Ch 11
    """
    t = np.atleast_1d(np.asarray(theta, dtype=float))
    if t.size == 0:
        raise ValueError("geron_l2_regularization: theta is empty")
    if not np.all(np.isfinite(t)):
        raise ValueError("geron_l2_regularization: theta contains non-finite values")
    a = float(alpha)
    if not np.isfinite(a) or a < 0:
        raise ValueError(f"geron_l2_regularization: alpha must be finite and non-negative, got {alpha!r}")

    mask = np.ones_like(t)
    if skip_bias:
        if t.size < 2:
            raise ValueError("geron_l2_regularization: skip_bias needs at least one non-bias parameter")
        mask.flat[0] = 0.0

    penalty = 0.5 * a * float(np.sum((t * mask) ** 2))
    grad = a * t * mask
    l2 = float(np.sqrt(np.sum((t * mask) ** 2)))

    shrink = None
    warns = []
    if eta is not None:
        lr = float(eta)
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError(f"geron_l2_regularization: eta must be a positive finite learning rate, got {eta!r}")
        shrink = 1.0 - lr * a
        if not (0.0 < shrink < 1.0):
            warns.append(
                f"weight decay factor 1 - eta*alpha = {shrink:g} is outside (0, 1); "
                f"the update overshoots or flips the sign of every weight."
            )

    return RichResult(
        title="L2 regularization",
        summary_lines=[("alpha", a), ("Penalty", penalty), ("||theta||_2", l2)],
        warnings=warns,
        interpretation="L2 shrinks every weight multiplicatively; it never zeroes one out, unlike L1.",
        payload={
            "penalty": penalty,
            "gradient": grad,
            "l2_norm": l2,
            "shrink_factor": shrink,
            "estimate": penalty,
            "n": int(t.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hml2r: L2 penalty (alpha/2)*sum theta^2 with gradient alpha*theta (weight decay)"
