# morie.fn -- function file (rootcoder007/morie)
"""Geometric extrapolation of the expected KDFE (Theorem 2.1)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["kdfgeoext", "fauzi_thm2_1_expected_kdfe"]


def kdfgeoext(jh, jah, a):
    r"""Geometric extrapolation of the expected KDFE (Theorem 2.1).

    Theorem 2.1: for any :math:`a>0`, :math:`a\ne1`,

    .. math:: [J_h(x)]^{t_1}[J_{ah}(x)]^{t_2} = F_X(x) + O(h^4),
              \qquad t_1=\frac{a^2}{a^2-1},\; t_2=-\frac1{a^2-1},

    with :math:`J_h = E[\hat F_h]`. The exponents solve
    :math:`t_1+t_2=1` (keep :math:`\log F_X`) and
    :math:`t_1+a^2t_2=0` (kill the :math:`h^2` term).

    Contrast Chapter 1, where the same device forced the bandwidth ratio
    to 4: there the expansion ran in :math:`\sqrt h` so the second
    condition was :math:`t_1+2t_2=0`, which pins the ratio. Here the
    expansion runs in :math:`h^2` and the ratio ``a`` stays FREE -- it is
    a genuine second smoothing parameter, and Remark 2.2 notes it need not
    depend on ``n`` at all. Large ``a`` returns the plain KDFE; ``a`` near
    0 is unwise because ``a`` divides the argument of ``W``.

    The exponents are returned so the caller can see they sum to 1; that
    is the invariant which makes the result a distribution function value
    rather than an arbitrary power.

    Parameters
    ----------
    jh : float
        ``J_h(x) = E[hat F_h(x)]``.
    jah : float
        ``J_{ah}(x)``, the same at bandwidth ``a*h``.
    a : float
        Second smoothing parameter; ``a > 0``, ``a != 1``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``t1``, ``t2``, ``a``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 2.1.
    """
    a = float(a)
    if a <= 0:
        raise ValueError(f"a must be positive, got {a}.")
    if a == 1.0:
        raise ValueError("a = 1 is excluded: the exponents divide by a^2 - 1.")
    jh = float(jh)
    jah = float(jah)
    if jh <= 0 or jah <= 0:
        raise ValueError("J_h and J_ah must be positive; the identity takes logs.")
    t1 = a * a / (a * a - 1.0)
    t2 = -1.0 / (a * a - 1.0)
    est = jh ** t1 * jah ** t2
    return RichResult(
        payload={
            "estimate": float(est),
            "t1": float(t1),
            "t2": float(t2),
            "a": a,
            "method": "geometric extrapolation of E[hat F_h] (Theorem 2.1)",
        }
    )


fauzi_thm2_1_expected_kdfe = kdfgeoext


def cheatsheet():
    return "fzt21: J_h^t1 J_ah^t2 = F + O(h^4) with a FREE bandwidth ratio a (Thm 2.1)"


# CANONICAL TEST
# >>> r = kdfgeoext(jh=0.5, jah=0.5, a=2.0)
# >>> abs(r['estimate'] - 0.5) < 1e-15 and abs(r['t1'] + r['t2'] - 1) < 1e-15
# True
