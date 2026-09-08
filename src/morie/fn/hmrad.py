# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reverse-mode automatic differentiation (backpropagation)."""

from . import _array_core as np

from ._richresult import RichResult
from .hmagrd import Var, geron_autograd

__all__ = ["geron_reverse_autodiff", "Var"]


def geron_reverse_autodiff(f, x):
    """
    Reverse-mode automatic differentiation (backprop) via the chain rule.

    Formula: build computation graph; propagate gradients backward

    The tape and the backward sweep are DELEGATED to the finished
    implementation :func:`morie.fn.hmagrd.geron_autograd`; this entry
    point is the differentiation-oriented one -- give it a scalar
    function and a point, get the gradient. ``f`` receives a list of
    :class:`~morie.fn.hmagrd.Var` leaves supporting ``+ - * / **`` and
    ``.exp() .log() .sqrt() .tanh() .sin() .cos() .relu() .sigmoid()``.

    One forward pass plus one backward pass yields ALL n partials, which
    is the whole reason reverse mode won: finite differences would need
    2n function evaluations for the same thing (compare
    :func:`~morie.fn.hmnmd.geron_numerical_diff`, which is the right tool
    for checking this one).

    Parameters
    ----------
    f : callable
        ``f(vars) -> Var``. A plain float return means the tape was
        bypassed and is rejected.
    x : array-like
        Point at which to differentiate.

    Returns
    -------
    result : RichResult
        Keys: gradient, value, tape_size, n_passes, estimate, n, method.

    Examples
    --------
    d(x0*x1)/dx = (x1, x0):

    >>> r = geron_reverse_autodiff(lambda v: v[0] * v[1], [3.0, 4.0])
    >>> float(r["value"]), [float(g) for g in r["gradient"]]
    (12.0, [4.0, 3.0])

    A composite: d/dx tanh(x)^2 at 0 is 0, and the value is 0 too:

    >>> r2 = geron_reverse_autodiff(lambda v: v[0].tanh() ** 2, [0.0])
    >>> float(r2["value"]), float(r2["gradient"][0])
    (0.0, 0.0)

    The whole gradient costs one backward pass whatever n is:

    >>> int(geron_reverse_autodiff(lambda v: v[0] + v[1] + v[2], [1.0, 2.0, 3.0])["n_passes"])
    2

    References
    ----------
    Geron Appendix A
    """
    if not callable(f):
        raise ValueError("geron_reverse_autodiff: f must be callable")
    base = geron_autograd(f, x)
    grad = np.asarray(base["grad"], dtype=float)
    return RichResult(
        title="Reverse-mode autodiff",
        summary_lines=[("Value", float(base["value"])), ("Tape nodes", int(base["tape_size"]))],
        interpretation="One forward and one backward pass give every partial, independent of the parameter count.",
        payload={
            "gradient": grad,
            "grad": grad,
            "value": float(base["value"]),
            "tape_size": int(base["tape_size"]),
            "n_passes": 2,
            "estimate": grad,
            "n": int(grad.size),
            "method": "Reverse-mode AD delegated to morie.fn.hmagrd.geron_autograd",
        },
    )


def cheatsheet():
    return "hmrad: Reverse-mode automatic differentiation (backprop)"
