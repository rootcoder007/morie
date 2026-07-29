# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Leaky ReLU: small negative slope prevents dead neurons."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_leaky_relu"]

_METHOD = "Leaky ReLU activation"


def geron_leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU: small negative slope prevents dead neurons.

    Formula: LReLU(z) = z if z>=0 else alpha*z

    The point of ``alpha > 0`` is that the derivative is never zero, so
    a unit pushed into the negative half-plane can still receive a
    gradient and come back -- a plain ReLU unit that lands there stops
    learning permanently.  The count of units in the leaky regime is
    returned, since that is the diagnostic the slope exists to fix.

    Parameters
    ----------
    z : array-like
        Pre-activations.
    alpha : float
        Negative slope, ``0 <= alpha < 1``.  ``alpha = 0`` recovers the
        plain ReLU and the dead-unit warning fires.

    Returns
    -------
    result : RichResult
        Keys: activation, derivative, n_leaky, alpha, estimate, n, method.

    Examples
    --------
    >>> r = geron_leaky_relu([-2.0, 0.0, 3.0], alpha=0.1)
    >>> [float(v) for v in r["activation"]]
    [-0.2, 0.0, 3.0]
    >>> [float(v) for v in r["derivative"]]
    [0.1, 1.0, 1.0]
    >>> r["n_leaky"]
    1

    The derivative matches a central difference away from the kink:

    >>> h = 1e-6
    >>> up = float(geron_leaky_relu([-2.0 + h], alpha=0.1)["activation"][0])
    >>> dn = float(geron_leaky_relu([-2.0 - h], alpha=0.1)["activation"][0])
    >>> round((up - dn) / (2 * h), 6)
    0.1

    References
    ----------
    Géron Ch 11
    """
    a = np.atleast_1d(np.asarray(z, dtype=float))
    if a.size == 0:
        raise ValueError("geron_leaky_relu: z is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("geron_leaky_relu: z contains non-finite values")
    slope = float(alpha)
    if not np.isfinite(slope) or slope < 0 or slope >= 1:
        raise ValueError(f"geron_leaky_relu: alpha must lie in [0, 1), got {alpha!r}")

    out = np.where(a >= 0, a, slope * a)
    deriv = np.where(a >= 0, 1.0, slope)
    n_leaky = int(np.count_nonzero(a < 0))

    return RichResult(
        title="Leaky ReLU",
        summary_lines=[("alpha", slope), ("Units in leaky regime", n_leaky), ("Units", int(a.size))],
        warnings=(
            ["alpha = 0 is a plain ReLU: units with negative pre-activation get zero gradient and can die."]
            if slope == 0
            else []
        ),
        payload={
            "activation": out,
            "derivative": deriv,
            "n_leaky": n_leaky,
            "alpha": slope,
            "estimate": float(np.mean(out)),
            "n": int(a.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlrel: Leaky ReLU max(alpha*z, z) with a never-zero derivative"
