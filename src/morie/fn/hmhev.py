# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Heaviside step activation function."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_heaviside"]

_METHOD = "Heaviside step activation"


def geron_heaviside(z, at_zero=1.0):
    """
    Heaviside step activation function.

    Formula: step(z) = 1 if z>=0 else 0

    The perceptron's activation.  Its derivative is zero everywhere it
    exists and undefined at the origin, which is exactly why gradient
    descent cannot train a network of these and why the sigmoid replaced
    it -- so the (a.e.) derivative is returned alongside as an all-zero
    array rather than left implicit.

    Parameters
    ----------
    z : array-like
        Pre-activations.
    at_zero : float
        Value assigned at ``z == 0``; the spec's convention is 1.

    Returns
    -------
    result : RichResult
        Keys: activation, derivative, n_active, estimate, n, method.

    Examples
    --------
    >>> r = geron_heaviside([-2.0, -0.0, 0.0, 0.5])
    >>> [float(v) for v in r["activation"]]
    [0.0, 1.0, 1.0, 1.0]
    >>> r["n_active"]
    3
    >>> [float(v) for v in r["derivative"]]
    [0.0, 0.0, 0.0, 0.0]
    >>> float(geron_heaviside([0.0], at_zero=0.0)["activation"][0])
    0.0

    References
    ----------
    Géron Ch 9
    """
    a = np.atleast_1d(np.asarray(z, dtype=float))
    if a.size == 0:
        raise ValueError("geron_heaviside: z is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("geron_heaviside: z contains non-finite values")
    tie = float(at_zero)
    if not np.isfinite(tie):
        raise ValueError(f"geron_heaviside: at_zero must be finite, got {at_zero!r}")

    out = np.where(a > 0, 1.0, np.where(a < 0, 0.0, tie))
    n_active = int(np.count_nonzero(out > 0))

    return RichResult(
        title="Heaviside step",
        summary_lines=[("Inputs", int(a.size)), ("Active units", n_active)],
        warnings=["The derivative is 0 almost everywhere, so this activation cannot be trained by backpropagation."],
        payload={
            "activation": out,
            "derivative": np.zeros_like(out),
            "n_active": n_active,
            "estimate": float(np.mean(out)),
            "n": int(a.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmhev: Heaviside step -- 1 if z >= 0 else 0; derivative 0 a.e. (untrainable)"
