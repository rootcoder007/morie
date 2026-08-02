# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Forward-mode autodiff via dual numbers."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_forward_mode_autodiff", "Dual"]

_METHOD = "Forward-mode autodiff (dual numbers)"


class Dual:
    r"""A dual number :math:`a + b\varepsilon` with :math:`\varepsilon^2 = 0`.

    Arithmetic on the pair carries the derivative along for free -- the
    product rule falls straight out of expanding
    :math:`(a + b\varepsilon)(c + d\varepsilon)` and dropping the
    :math:`\varepsilon^2` term.

    Supports ``+ - * / **`` with dual numbers and plain floats, plus
    ``exp``, ``log``, ``sqrt``, ``sin``, ``cos`` and ``tanh`` as
    methods.
    """

    __slots__ = ("value", "deriv")

    def __init__(self, value, deriv=0.0):
        self.value = float(value)
        self.deriv = float(deriv)

    @staticmethod
    def _lift(o):
        return o if isinstance(o, Dual) else Dual(o, 0.0)

    def __repr__(self):
        return f"Dual({self.value}, {self.deriv})"

    def __add__(self, o):
        o = self._lift(o)
        return Dual(self.value + o.value, self.deriv + o.deriv)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.deriv)

    def __sub__(self, o):
        return self + (-self._lift(o))

    def __rsub__(self, o):
        return self._lift(o) + (-self)

    def __mul__(self, o):
        o = self._lift(o)
        return Dual(self.value * o.value,
                    self.deriv * o.value + self.value * o.deriv)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._lift(o)
        if o.value == 0:
            raise ValueError("division by a dual number with value 0 is undefined.")
        return Dual(self.value / o.value,
                    (self.deriv * o.value - self.value * o.deriv) / (o.value**2))

    def __rtruediv__(self, o):
        return self._lift(o) / self

    def __pow__(self, p):
        p = float(p)
        return Dual(self.value**p, p * self.value ** (p - 1.0) * self.deriv)

    def exp(self):
        e = np.exp(self.value)
        return Dual(e, e * self.deriv)

    def log(self):
        if self.value <= 0:
            raise ValueError(f"log is undefined at {self.value}.")
        return Dual(np.log(self.value), self.deriv / self.value)

    def sqrt(self):
        if self.value <= 0:
            raise ValueError(f"sqrt derivative is undefined at {self.value}.")
        s = np.sqrt(self.value)
        return Dual(s, self.deriv / (2.0 * s))

    def sin(self):
        return Dual(np.sin(self.value), np.cos(self.value) * self.deriv)

    def cos(self):
        return Dual(np.cos(self.value), -np.sin(self.value) * self.deriv)

    def tanh(self):
        t = np.tanh(self.value)
        return Dual(t, (1.0 - t * t) * self.deriv)


def geron_forward_mode_autodiff(x, x_prime, f):
    r"""Evaluate ``f`` and its directional derivative in one pass.

    .. math::
        f(x + x'\varepsilon) = f(x) + f'(x)\,x'\,\varepsilon,
        \qquad \varepsilon^2 = 0

    Because :math:`\varepsilon^2 = 0` every higher-order term vanishes
    on its own -- the derivative is *exact*, not a finite difference,
    and there is no step size to choose.  A central difference is
    computed alongside and reported as ``finite_difference_check``, and
    the two agreeing to a few decimals is the sanity check; the dual
    number is the one to believe.

    Forward mode costs one pass *per input direction*, which is why it
    is the wrong tool for a neural net (millions of parameters, one
    loss) and the right one for the opposite shape.  Reverse mode --
    :func:`morie.fn.graut.geron_autograd_chain_rule` -- is the mirror
    image.

    ``f`` must be written in terms of the arithmetic operators and the
    :class:`Dual` methods (``.exp()``, ``.log()``, ``.sin()`` ...), not
    ``numpy.exp``.

    Parameters
    ----------
    x : float
        Point of evaluation.
    x_prime : float
        Seed direction; ``1.0`` gives the ordinary derivative.
    f : callable
        ``f(Dual) -> Dual``.

    Returns
    -------
    RichResult
        Payload keys ``value``, ``derivative``, ``finite_difference_check``,
        ``check_abs_error``, ``x``, ``x_prime``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Appendix A, Forward-mode autodiff section.

    Examples
    --------
    :math:`f(z) = z^3` at ``z = 2``: value 8, derivative
    :math:`3z^2 = 12`, with no truncation error at all.

    >>> r = geron_forward_mode_autodiff(2.0, 1.0, lambda z: z * z * z)
    >>> r["value"], r["derivative"]
    (8.0, 12.0)
    >>> r["check_abs_error"] < 1e-6
    True

    The seed scales the derivative -- that is what makes it
    *directional*:

    >>> geron_forward_mode_autodiff(2.0, 3.0, lambda z: z * z)["derivative"]
    12.0

    Transcendentals work through the Dual methods; :math:`e^z` is its
    own derivative:

    >>> r2 = geron_forward_mode_autodiff(0.0, 1.0, lambda z: z.exp())
    >>> r2["value"], r2["derivative"]
    (1.0, 1.0)

    Géron's Appendix A example, :math:`f(x, y) = x^2 y + y + 2` in the
    ``x`` direction at ``(3, 4)``: :math:`\partial f/\partial x = 2xy =
    24`.

    >>> r3 = geron_forward_mode_autodiff(3.0, 1.0, lambda z: z * z * 4.0 + 4.0 + 2.0)
    >>> r3["value"], r3["derivative"]
    (42.0, 24.0)
    """
    if not callable(f):
        raise ValueError(f"f must be callable, got {type(f).__name__}.")
    x = float(x)
    x_prime = float(x_prime)
    if not np.isfinite(x) or not np.isfinite(x_prime):
        raise ValueError(f"x and x_prime must be finite, got {x}, {x_prime}.")

    out = f(Dual(x, x_prime))
    if not isinstance(out, Dual):
        raise ValueError(
            f"f must return a Dual; it returned {type(out).__name__}. Write f with "
            f"the arithmetic operators and Dual methods (z.exp(), z.log(), ...), "
            f"not numpy functions."
        )
    if not np.isfinite(out.value) or not np.isfinite(out.deriv):
        raise ValueError("f produced a non-finite value or derivative.")

    h = 1e-6 * max(1.0, abs(x))
    try:
        fd = (f(Dual(x + h, 0.0)).value - f(Dual(x - h, 0.0)).value) / (2.0 * h) * x_prime
        err = float(abs(fd - out.deriv))
    except (ValueError, ZeroDivisionError):
        fd, err = None, None

    return RichResult(
        title="Forward-mode autodiff",
        summary_lines=[("f(x)", out.value), ("f'(x) * x'", out.deriv)],
        payload={
            "value": float(out.value),
            "derivative": float(out.deriv),
            "finite_difference_check": fd,
            "check_abs_error": err,
            "x": x,
            "x_prime": x_prime,
            "estimate": float(out.deriv),
            "n": 1,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grfad: f(x + x'eps) with eps^2 = 0 gives the exact derivative in one pass, no step size"
