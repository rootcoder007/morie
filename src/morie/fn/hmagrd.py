# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Automatic differentiation via reverse-mode autograd."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_autograd", "Var"]


class Var:
    """Scalar node on a reverse-mode tape.

    Each node stores its value and the list of (parent, local derivative)
    pairs; because every operation is eager and scalar, the local partials
    are plain floats known at construction time.
    """

    __slots__ = ("value", "grad", "parents", "op")

    def __init__(self, value, parents=(), op="leaf"):
        self.value = float(value)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.op = op

    def __repr__(self):
        return f"Var({self.value!r}, op={self.op!r})"

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Var) else Var(x)

    def __add__(self, other):
        o = Var._wrap(other)
        return Var(self.value + o.value, ((self, 1.0), (o, 1.0)), "add")

    __radd__ = __add__

    def __neg__(self):
        return Var(-self.value, ((self, -1.0),), "neg")

    def __sub__(self, other):
        o = Var._wrap(other)
        return Var(self.value - o.value, ((self, 1.0), (o, -1.0)), "sub")

    def __rsub__(self, other):
        return Var._wrap(other) - self

    def __mul__(self, other):
        o = Var._wrap(other)
        return Var(self.value * o.value, ((self, o.value), (o, self.value)), "mul")

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = Var._wrap(other)
        if o.value == 0.0:
            raise ValueError("Var.__truediv__: division by zero on the tape")
        return Var(
            self.value / o.value,
            ((self, 1.0 / o.value), (o, -self.value / (o.value * o.value))),
            "div",
        )

    def __rtruediv__(self, other):
        return Var._wrap(other) / self

    def __pow__(self, p):
        c = float(p)
        if self.value == 0.0 and c < 1.0:
            raise ValueError(f"Var.__pow__: derivative of x**{c} is undefined at x=0")
        if self.value < 0.0 and c != int(c):
            raise ValueError("Var.__pow__: negative base with fractional exponent is not real")
        return Var(self.value**c, ((self, c * self.value ** (c - 1.0)),), "pow")

    def exp(self):
        e = math.exp(self.value)
        return Var(e, ((self, e),), "exp")

    def log(self):
        if self.value <= 0.0:
            raise ValueError(f"Var.log: log is undefined at {self.value}")
        return Var(math.log(self.value), ((self, 1.0 / self.value),), "log")

    def sqrt(self):
        if self.value <= 0.0:
            raise ValueError(f"Var.sqrt: derivative of sqrt is undefined at {self.value}")
        s = math.sqrt(self.value)
        return Var(s, ((self, 0.5 / s),), "sqrt")

    def tanh(self):
        t = math.tanh(self.value)
        return Var(t, ((self, 1.0 - t * t),), "tanh")

    def sin(self):
        return Var(math.sin(self.value), ((self, math.cos(self.value)),), "sin")

    def cos(self):
        return Var(math.cos(self.value), ((self, -math.sin(self.value)),), "cos")

    def relu(self):
        return Var(max(self.value, 0.0), ((self, 1.0 if self.value > 0.0 else 0.0),), "relu")

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.value))
        return Var(s, ((self, s * (1.0 - s)),), "sigmoid")

    def backward(self):
        """Seed d(self)/d(self) = 1 and propagate in reverse topological order."""
        order = []
        seen = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p, _ in node.parents:
                if id(p) not in seen:
                    stack.append((p, False))
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            g = node.grad
            if g == 0.0:
                continue
            for p, local in node.parents:
                p.grad += g * local
        return order


def geron_autograd(loss, params):
    """
    Automatic differentiation via reverse-mode autograd.

    Formula: loss.backward() populates .grad on leaf tensors

    `loss` is a callable that receives a list of :class:`Var` leaves and
    returns a single :class:`Var` -- the scalar to differentiate. Vars
    support ``+ - * / **`` and ``.exp() .log() .sqrt() .tanh() .sin()
    .cos() .relu() .sigmoid()``. One backward sweep fills every leaf's
    ``.grad`` with the exact partial derivative, at O(cost of the forward
    pass), which is what makes reverse mode worth having.

    Parameters
    ----------
    loss : callable
        ``loss(vars) -> Var``. Returning anything else is an error: a plain
        float means the tape was bypassed and no gradient exists.
    params : array-like
        Values of the leaf parameters (1-D).

    Returns
    -------
    result : RichResult
        Keys: grad, value, tape_size, estimate, n, method.

    Examples
    --------
    >>> f = lambda p: p[0] * p[1] + p[0].exp()
    >>> r = geron_autograd(f, [0.0, 3.0])
    >>> float(r["value"])
    1.0
    >>> [float(g) for g in r["grad"]]
    [4.0, 0.0]
    >>> r2 = geron_autograd(lambda p: (p[0] ** 2 + 1.0).log(), [2.0])
    >>> round(float(r2["value"]), 9), round(float(r2["grad"][0]), 9)
    (1.609437912, 0.8)

    References
    ----------
    Géron Ch 10
    """
    if not callable(loss):
        raise ValueError("geron_autograd: loss must be callable")
    p = np.atleast_1d(np.asarray(params, dtype=float)).ravel()
    if p.size == 0:
        raise ValueError("geron_autograd: params is empty")
    if not np.all(np.isfinite(p)):
        raise ValueError("geron_autograd: params must be finite")

    leaves = [Var(float(x)) for x in p]
    out = loss(leaves)
    if not isinstance(out, Var):
        raise ValueError(
            f"geron_autograd: loss must return a Var built from the supplied leaves, got {type(out).__name__}; "
            "a raw float means the tape was bypassed and no gradient can be recovered"
        )
    order = out.backward()
    grad = np.array([leaf.grad for leaf in leaves], dtype=float)

    return RichResult(
        title="Reverse-mode autograd",
        summary_lines=[("Loss", out.value), ("Gradient norm", float(np.linalg.norm(grad))), ("Tape nodes", len(order))],
        payload={
            "grad": grad,
            "value": out.value,
            "tape_size": len(order),
            "params": p,
            "estimate": out.value,
            "n": int(p.size),
            "method": "Reverse-mode automatic differentiation over a scalar tape",
        },
    )


def cheatsheet():
    return "hmagrd: Automatic differentiation via reverse-mode autograd"


# compact alias per ledger/NAMING.md
geronautograd = geron_autograd
