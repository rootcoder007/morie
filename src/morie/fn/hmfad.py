# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Forward-mode automatic differentiation via dual numbers."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_forward_autodiff", "Dual"]


class Dual:
    """A dual number ``a + b eps`` with ``eps^2 = 0``.

    Arithmetic on ``Dual`` carries the derivative alongside the value, so
    evaluating ``f`` once at ``a + 1*eps`` yields ``f(a) + f'(a)*eps``.
    Nothing is approximated: the chain rule is applied exactly by the
    operator overloads.
    """

    __slots__ = ("value", "deriv")

    def __init__(self, value, deriv=0.0):
        self.value = float(value)
        self.deriv = float(deriv)

    @staticmethod
    def _c(o):
        return o if isinstance(o, Dual) else Dual(o, 0.0)

    def __repr__(self):
        return f"Dual({self.value}, {self.deriv})"

    def __add__(self, o):
        o = Dual._c(o)
        return Dual(self.value + o.value, self.deriv + o.deriv)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.deriv)

    def __sub__(self, o):
        return self + (-Dual._c(o))

    def __rsub__(self, o):
        return Dual._c(o) + (-self)

    def __mul__(self, o):
        o = Dual._c(o)
        return Dual(self.value * o.value, self.deriv * o.value + self.value * o.deriv)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Dual._c(o)
        if o.value == 0:
            raise ValueError("Dual: division by zero")
        return Dual(self.value / o.value, (self.deriv * o.value - self.value * o.deriv) / (o.value**2))

    def __rtruediv__(self, o):
        return Dual._c(o) / self

    def __pow__(self, p):
        p = float(p)
        if self.value == 0 and p < 1:
            raise ValueError(f"Dual: x**{p} is not differentiable at x = 0")
        return Dual(self.value**p, p * self.value ** (p - 1) * self.deriv)

    def exp(self):
        e = math.exp(self.value)
        return Dual(e, e * self.deriv)

    def log(self):
        if self.value <= 0:
            raise ValueError(f"Dual: log requires a positive argument, got {self.value}")
        return Dual(math.log(self.value), self.deriv / self.value)

    def sin(self):
        return Dual(math.sin(self.value), math.cos(self.value) * self.deriv)

    def cos(self):
        return Dual(math.cos(self.value), -math.sin(self.value) * self.deriv)

    def sqrt(self):
        if self.value <= 0:
            raise ValueError(f"Dual: sqrt is not differentiable at {self.value}")
        s = math.sqrt(self.value)
        return Dual(s, self.deriv / (2 * s))

    def tanh(self):
        t = math.tanh(self.value)
        return Dual(t, (1 - t * t) * self.deriv)


def geron_forward_autodiff(f, x):
    """
    Forward-mode automatic differentiation via dual numbers.

    Formula: d/dx f(x) computed by forward propagation of dual components

    ``f`` is evaluated once per input coordinate with that coordinate
    seeded as ``x_i + 1*eps`` and the rest at ``eps = 0``, so the cost is
    ``n`` passes for ``n`` inputs -- the reason forward mode is the wrong
    choice for training a network (many inputs, one output) and the right
    one for the opposite shape.

    The result is exact to machine precision, not a difference quotient;
    ``fd_check`` runs a central finite difference alongside it so the two
    can be compared, and their gap is reported.

    ``f`` always receives a *list* of :class:`Dual` (length 1 for a scalar
    input) and must return a ``Dual``; returning a float means the
    dual thread was broken -- e.g. by calling ``math.exp`` instead of the
    ``.exp()`` method -- and raises.

    Parameters
    ----------
    f : callable
        ``f(duals) -> Dual``.
    x : float or array-like
        Point at which to differentiate.

    Returns
    -------
    result : RichResult
        Keys: value, grad, n_passes, fd_check, max_fd_gap, estimate,
        n, method.

    Examples
    --------
    ``f(x) = x^2`` has derivative ``2x``:

    >>> r = geron_forward_autodiff(lambda v: v[0] ** 2, [3.0])
    >>> r["value"], r["grad"]
    (9.0, [6.0])

    A two-input function, matching the tape-based ``hmagrd`` example
    ``f(x, y) = x*y + exp(x)`` at ``(0, 3)``:

    >>> r2 = geron_forward_autodiff(lambda v: v[0] * v[1] + v[0].exp(), [0.0, 3.0])
    >>> r2["value"], r2["grad"]
    (1.0, [4.0, 0.0])
    >>> r2["n_passes"]
    2

    ``log(x^2 + 1)`` at ``x = 2`` has derivative ``4/5``:

    >>> r3 = geron_forward_autodiff(lambda v: (v[0] ** 2 + 1.0).log(), 2.0)
    >>> round(r3["value"], 9), round(r3["grad"][0], 9)
    (1.609437912, 0.8)
    >>> r3["max_fd_gap"] < 1e-6
    True

    References
    ----------
    Géron Appendix A
    """
    if not callable(f):
        raise ValueError("geron_forward_autodiff: f must be callable")
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    if xs.size == 0:
        raise ValueError("geron_forward_autodiff: x is empty")
    if not np.all(np.isfinite(xs)):
        raise ValueError("geron_forward_autodiff: x contains non-finite values")
    n = xs.size

    def call(vals, seed_index):
        duals = [Dual(vals[i], 1.0 if i == seed_index else 0.0) for i in range(n)]
        out = f(duals)
        if not isinstance(out, Dual):
            raise ValueError(
                f"geron_forward_autodiff: f returned {type(out).__name__}, not a Dual -- "
                "use the Dual methods (.exp(), .log(), .sin()) so the derivative thread survives"
            )
        return out

    grad = np.empty(n)
    value = None
    for i in range(n):
        out = call(xs, i)
        grad[i] = out.deriv
        value = out.value

    # Independent central finite difference for comparison.
    def plain(vals):
        out = f([Dual(v, 0.0) for v in vals])
        return out.value

    h = 1e-5
    fd = np.empty(n)
    for i in range(n):
        up, dn = xs.copy(), xs.copy()
        up[i] += h
        dn[i] -= h
        fd[i] = (plain(up) - plain(dn)) / (2 * h)

    gap = float(np.max(np.abs(fd - grad)))

    return RichResult(
        title="Forward-mode autodiff",
        summary_lines=[("Value", value), ("Passes", n)],
        interpretation="Forward mode costs one pass per input, so it is cheap only when inputs are few.",
        payload={
            "value": float(value),
            "grad": grad.tolist(),
            "gradient": grad.tolist(),
            "n_passes": int(n),
            "fd_check": fd.tolist(),
            "max_fd_gap": gap,
            "estimate": float(np.linalg.norm(grad)),
            "n": int(n),
            "method": "forward-mode autodiff with dual numbers (exact chain rule)",
        },
    )


def cheatsheet():
    return "hmfad: Forward-mode automatic differentiation via dual numbers"
