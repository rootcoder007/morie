# morie.fn -- function file (rootcoder007/morie)
"""The c_1 bias coefficient of the boundary-free KDFE (Eq. 5.8)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bfc1", "fauzi_c1_coefficient"]


def bfc1(dg, d2g, density, fp):
    r"""The c_1 bias coefficient of the boundary-free KDFE (Eq. 5.8).

    Eq. (5.8):

    .. math:: c_1(x) = g''(g^{-1}(x))f_X(x)
              + [g'(g^{-1}(x))]^2 f_X'(x),

    the coefficient in :math:`\mathrm{Bias}[\tilde F_X(x)]
    = \tfrac{h^2}2 c_1(x)\mu_2(K) + o(h^2)` of Theorem 5.2.

    Compare the naive KDFE, whose bias coefficient is just
    :math:`f_X'(x)/2\cdot\mu_2` -- i.e. (5.8) with
    :math:`g = \mathrm{identity}`, where :math:`g''=0` and
    :math:`g'=1`. The transformation adds one term and rescales the
    other; that is the entire cost of removing the boundary bias.

    This is the same expression as :math:`b_1(t)` in (4.14) of Chapter 4.
    The survival estimator and the distribution estimator are the same
    construction applied to :math:`1-F` and :math:`F`, so they share a
    bias coefficient -- and Theorem 4.1 duly carries a minus sign in front
    of it where Theorem 5.2 does not.

    Parameters
    ----------
    dg, d2g : float
        ``g'(g^{-1}(x))`` and ``g''(g^{-1}(x))``.
    density : float
        ``f_X(x)``.
    fp : float
        ``f_X'(x)``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (5.8); the same expression as (4.14).
    """
    val = float(d2g) * float(density) + float(dg) ** 2 * float(fp)
    return RichResult(
        payload={
            "estimate": float(val),
            "method": "c_1 bias coefficient of the boundary-free KDFE (Eq. 5.8)",
        }
    )


fauzi_c1_coefficient = bfc1


def cheatsheet():
    return "fzc1x: c_1 = g'' f + (g')^2 f'; with g = identity it collapses to the naive f' (5.8)"


# CANONICAL TEST
# >>> abs(bfc1(dg=1.0, d2g=0.0, density=0.3, fp=0.2)['estimate'] - 0.2) < 1e-15
# True
