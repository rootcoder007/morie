# morie.fn -- function file (rootcoder007/morie)
"""The c_2 bias coefficient of the boundary-free KDE (Theorem 5.5)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bfc2", "fauzi_c2_coefficient"]


def bfc2(dg, d2g, d3g, density, fp, fpp):
    r"""The c_2 bias coefficient of the boundary-free KDE (Theorem 5.5).

    From Theorem 5.5:

    .. math:: c_2(x) = g^{(3)}(g^{-1}(x))f_X(x)
              + 3g''(g^{-1}(x))g'(g^{-1}(x))f_X'(x)
              + [g'(g^{-1}(x))]^3 f_X''(x).

    The three coefficients 1, 3, 1 and the derivative orders are the
    Faa di Bruno pattern for the second derivative of a composition --
    which is what this is, since the estimator lives on the transformed
    scale and the bias is read back on the original one.

    It enters the bias of the boundary-free DENSITY estimator divided by
    :math:`g'`:
    :math:`\mathrm{Bias}[\tilde f_X(x)] =
    h^2c_2(x)\mu_2(K)/(2g'(g^{-1}(x))) + o(h^2)`.
    That division is the Jacobian which the DISTRIBUTION estimator of
    (5.5) does not need -- the clearest statement in the book of why the
    transformation trick is cheaper for distribution functions.

    With ``g`` the identity (:math:`g'=1`, :math:`g''=g^{(3)}=0`) this
    collapses to :math:`f_X''(x)`, the classical kernel-density bias
    coefficient.

    Parameters
    ----------
    dg, d2g, d3g : float
        ``g'``, ``g''`` and ``g^(3)`` evaluated at ``g^{-1}(x)``.
    density : float
        ``f_X(x)``.
    fp, fpp : float
        ``f_X'(x)`` and ``f_X''(x)``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``scaled``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 5.5.
    """
    dg = float(dg)
    val = (
        float(d3g) * float(density)
        + 3.0 * float(d2g) * dg * float(fp)
        + dg ** 3 * float(fpp)
    )
    if dg == 0.0:
        raise ValueError("g'(g^-1(x)) must be non-zero; the bias divides by it.")
    return RichResult(
        payload={
            "estimate": float(val),
            "scaled": float(val / dg),
            "method": "c_2 bias coefficient of the boundary-free KDE (Theorem 5.5)",
        }
    )


fauzi_c2_coefficient = bfc2


def cheatsheet():
    return "fzc2x: c_2: the Faa di Bruno 1-3-1 pattern; the DENSITY bias needs a Jacobian, the df does not"


# CANONICAL TEST
# >>> r = bfc2(dg=1.0, d2g=0.0, d3g=0.0, density=0.3, fp=0.2, fpp=-0.1)
# >>> abs(r['estimate'] + 0.1) < 1e-15
# True
