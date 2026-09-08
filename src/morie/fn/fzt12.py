# morie.fn -- function file (rootcoder007/morie)
"""Geometric extrapolation of the raw gamma-kernel mean (Theorem 1.2)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["gkgeoext", "fauzi_thm1_2_var_mgkde"]


def gkgeoext(jh, j4h, h=None, a=None, b=None, f=None):
    r"""Geometric extrapolation of the raw gamma-kernel mean (Theorem 1.2).

    Theorem 1.2, Eq. (1.13):

    .. math:: J_h^2(x)\,[J_{4h}(x)]^{-1} = f_X(x) + O(h),

    with :math:`J_h = E[A_h]`. The exponents are not a guess: writing
    :math:`\log J_h = \log f + a\sqrt h/f + (b - a^2/2f)h/f`, the pair
    :math:`(t_1,t_2)` must satisfy :math:`t_1+t_2=1` (keep
    :math:`\log f`) and :math:`t_1+2t_2=0` (kill :math:`\sqrt h`); the
    unique solution is :math:`(2,-1)`. That is the whole trick -- the
    O(sqrt h) bias of Theorem 1.1 is cancelled by a WEIGHTED GEOMETRIC
    mean of two smoothings, not an arithmetic one, because the expansion
    is multiplicative in ``f``.

    Supply ``a``, ``b`` and ``f`` to also get the explicit O(h) remainder
    :math:`-2(b - a^2/2f)h`; without them ``remainder`` is NaN and the
    docstring says so rather than inventing a value.

    Parameters
    ----------
    jh : float
        ``J_h(x) = E[A_h(x)]``.
    j4h : float
        ``J_{4h}(x) = E[A_{4h}(x)]``, the same object at bandwidth ``4h``.
    h : float, optional
        Bandwidth, needed only for the explicit remainder.
    a, b, f : float, optional
        The coefficients (1.16), (1.17) and ``f_X(x)``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``remainder``, ``t1``, ``t2``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 1.2, Eq. (1.13).
    """
    jh = float(jh)
    j4h = float(j4h)
    if j4h == 0.0:
        raise ValueError("J_4h(x) must be non-zero -- (1.13) divides by it.")
    est = jh * jh / j4h
    rem = np.nan
    if None not in (h, a, b, f):
        if float(f) == 0.0:
            raise ValueError("f_X(x) must be non-zero in the (1.13) remainder.")
        rem = -2.0 * (float(b) - float(a) ** 2 / (2.0 * float(f))) * float(h)
    return RichResult(
        payload={
            "estimate": float(est),
            "remainder": float(rem),
            "t1": 2.0,
            "t2": -1.0,
            "method": "geometric extrapolation of E[A_h] (Theorem 1.2)",
        }
    )


fauzi_thm1_2_var_mgkde = gkgeoext


def cheatsheet():
    return "fzt12: geometric extrapolation J_h^2/J_4h kills the O(sqrt h) bias (Thm 1.2)"


# CANONICAL TEST
# >>> r = gkgeoext(jh=0.4, j4h=0.5)
# >>> abs(r['estimate'] - 0.32) < 1e-15
# True
