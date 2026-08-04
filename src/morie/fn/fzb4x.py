# morie.fn -- function file (rootcoder007/morie)
"""The b_4 bias coefficient of the kernel distribution function estimator."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["kdfb4", "fauzi_b4_coefficient"]


def kdfb4(fppp, mu4=3.0):
    r"""The b_4 bias coefficient of the kernel distribution function estimator.

    Eq. (2.8):

    .. math:: b_4(x) = \frac{f_X^{(3)}(x)}{24}
              \int_{-\infty}^{\infty}w^4K(w)\,dw.

    The second bias coefficient in the expansion
    :math:`J_h(x) = F_X(x)(1 + h^2b_2/F_X + h^4b_4/F_X) + o(h^4)`. It only
    matters once :math:`b_2` has been eliminated -- which is exactly what
    the geometric extrapolation of Theorem 2.1 does, leaving
    :math:`h^4a^2(b_2^2 - 2b_4F_X)/(2F_X)` as the whole bias.

    So :math:`b_4` is not a refinement of the standard KDFE's error; it IS
    the error of the bias-reduced one. Assumption B2 (finite
    :math:`\mu_4(K)`) and B4 (:math:`f_X^{(4)}` exists) are needed for
    precisely this term.

    ``mu4`` defaults to 3, the Gaussian value.

    Parameters
    ----------
    fppp : float
        The third derivative ``f_X^(3)(x)``.
    mu4 : float, default 3.0
        ``int w^4 K(w) dw``; 3 for the Gaussian kernel.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``mu4``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (2.8).
    """
    val = float(fppp) / 24.0 * float(mu4)
    return RichResult(
        payload={
            "estimate": float(val),
            "mu4": float(mu4),
            "method": "b_4 bias coefficient of the KDFE (Eq. 2.8)",
        }
    )


fauzi_b4_coefficient = kdfb4


def cheatsheet():
    return "fzb4x: b_4(x) = f^(3)(x) mu_4(K) / 24 -- the bias of the EXTRAPOLATED KDFE (Eq. 2.8)"


# CANONICAL TEST
# >>> abs(kdfb4(fppp=24.0)['estimate'] - 3.0) < 1e-15
# True
