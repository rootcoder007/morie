# morie.fn -- function file (rootcoder007/morie)
"""Bias of the bias-reduced KDFE (Theorem 2.2)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["gekdfbias", "fauzi_thm2_2_bias_brdkdfe"]


def gekdfbias(h, a, b2, b4, fx):
    r"""Bias of the bias-reduced KDFE (Theorem 2.2).

    Theorem 2.2, Eq. (2.6):

    .. math:: \mathrm{Bias}[\tilde F_X(x)] = h^4a^2
              \frac{b_2^2(x) - 2b_4(x)F_X(x)}{2F_X(x)} + o(h^4)
              + O(n^{-1}),

    with :math:`b_2` from (2.7) and :math:`b_4` from (2.8).

    Two things worth naming. First the :math:`O(n^{-1})` term: it is not a
    smoothing error at all but the price of the estimator being a
    NONLINEAR function of two linear statistics, coming from the
    :math:`O(p^2)` remainder in :math:`(1+p)^q = 1+pq+O(p^2)`. So there is
    a floor on the bias that no bandwidth can cross.

    Second the :math:`a^2` factor: the bias GROWS with the second
    smoothing parameter, while Remark 2.2 says :math:`\tilde F_X \to \hat
    F_h` as :math:`a\to\infty`. Both are true -- the :math:`h^4a^2` form
    is an asymptotic statement in ``h`` for FIXED ``a``, and Table 2.1
    duly shows the smallest AISE at :math:`a = 0.01`.

    Parameters
    ----------
    h : float
        Bandwidth.
    a : float
        Second smoothing parameter; ``a > 0``, ``a != 1``.
    b2, b4 : float
        The coefficients (2.7) and (2.8) at ``x``.
    fx : float
        ``F_X(x)``, strictly between 0 and 1.

    Returns
    -------
    RichResult
        Keys ``bias``, ``leading``, ``h``, ``a``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 2.2, Eqs. (2.6)-(2.8).
    """
    h = float(h)
    a = float(a)
    fx = float(fx)
    if a <= 0 or a == 1.0:
        raise ValueError("a must be positive and different from 1.")
    if not 0.0 < fx < 1.0:
        raise ValueError(f"F_X(x) must lie strictly in (0, 1), got {fx}.")
    lead = (float(b2) ** 2 - 2.0 * float(b4) * fx) / (2.0 * fx)
    return RichResult(
        payload={
            "bias": float(h ** 4 * a * a * lead),
            "leading": float(lead),
            "h": h,
            "a": a,
            "method": "bias of the bias-reduced KDFE (Theorem 2.2)",
        }
    )


fauzi_thm2_2_bias_brdkdfe = gekdfbias


def cheatsheet():
    return "fzt22: bias h^4 a^2 (b2^2 - 2 b4 F)/(2F), plus an O(1/n) floor from nonlinearity (Thm 2.2)"


# CANONICAL TEST
# >>> r = gekdfbias(h=0.1, a=2.0, b2=0.0, b4=0.5, fx=0.5)
# >>> abs(r['bias'] - 1e-4 * 4 * (-0.5)) < 1e-18
# True
