# morie.fn -- function file (rootcoder007/morie)
"""Formal derivative of a polynomial.

Classical algebra.  Triage confirmed this names no owning source.

Coefficient order: lowest degree first, so ``poly[i]`` multiplies
x**i.  The same convention is used by :mod:`morie.fn.resaln`.
"""

from ._richresult import RichResult, with_describe_pointer

__all__ = ["formal_derivative"]


def formal_derivative(poly):
    """Differentiate sum_i a_i x**i term by term,

        d/dx sum_i a_i x**i = sum_i i a_i x**(i-1).

    "Formal" because nothing analytic is used: the rule is applied to
    the coefficient list itself, so it is equally valid over a ring
    where limits make no sense.  The derivative of a constant is the
    zero polynomial, represented as ``[0.0]``.

    Parameters
    ----------
    poly : sequence of coefficients, lowest degree first.

    Returns
    -------
    RichResult with keys estimate (the leading coefficient of the
    derivative), coefficients, degree, method.
    """
    a = [float(v) for v in poly]
    if not a:
        raise ValueError("polynomial needs at least one coefficient")
    d = [i * a[i] for i in range(1, len(a))]
    if not d:
        d = [0.0]
    deg = len(d) - 1
    while deg > 0 and d[deg] == 0.0:
        deg -= 1
    return with_describe_pointer(RichResult(payload={
        "estimate": float(d[deg]), "coefficients": d, "degree": deg,
        "method": "formal derivative of a polynomial",
    }), "frmlD")


def cheatsheet():
    return "frmlD: Formal derivative of polynomial"


# compact alias per ledger/NAMING.md
formalderiv = formal_derivative
