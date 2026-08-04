# morie.fn -- function file (rootcoder007/morie)
"""The b_5 coefficient of the mean-residual-life variance (Eq. 4.28)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["mrlb5", "fauzi_b5_coefficient_mrl"]


def mrlb5(dg, density, mrl, surv=None):
    r"""The b_5 coefficient of the mean-residual-life variance (Eq. 4.28).

    Eq. (4.28), second half:

    .. math:: b_5(t) = g'(g^{-1}(t))f_X(t)m_X^2(t).

    It multiplies the :math:`-h/n` term of (4.27), so it is the SIZE OF
    THE GAIN from smoothing:

    .. math:: \mathrm{Var}[m_{X,i}(t)] = \frac1n\frac{b_4(t)}{S_X^2(t)}
              - \frac hn\frac{b_5(t)}{S_X^2(t)}\!\int\! V(y)W(y)dy
              + o\!\big(\tfrac hn\big).

    The sign is the same lesson as :math:`r_1` in Chapter 2 -- smoothing a
    distribution-type functional REDUCES variance -- and the same reason
    the bandwidth rate here is :math:`n^{-1/3}`, not :math:`n^{-1/5}`.

    Note where :math:`g'` enters: the gain is proportional to the
    derivative of the transformation at the point. A transformation that
    stretches near :math:`t` buys more variance reduction there, which is
    the mechanism behind Remark 5.1's observation that
    :math:`g'(g^{-1}(x)) \ge 1` makes the boundary-free estimator strictly
    better than the naive one.

    Parameters
    ----------
    dg : float
        ``g'(g^{-1}(t))``.
    density : float
        ``f_X(t)``.
    mrl : float
        ``m_X(t)``.
    surv : float, optional
        ``S_X(t)``; needed only for ``varterm``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``varterm``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eqs. (4.27)-(4.28).
    """
    val = float(dg) * float(density) * float(mrl) ** 2
    if surv is None:
        varterm = np.nan
    else:
        s = float(surv)
        if s <= 0:
            raise ValueError(f"S_X(t) must be positive, got {s}.")
        varterm = val / (s * s)
    return RichResult(
        payload={
            "estimate": float(val),
            "varterm": float(varterm),
            "method": "b_5 coefficient of the MRL variance (Eq. 4.28)",
        }
    )


fauzi_b5_coefficient_mrl = mrlb5


def cheatsheet():
    return "fzb5t: b_5 = g' f m^2 -- the size of the variance GAIN from smoothing (4.28)"


# CANONICAL TEST
# >>> abs(mrlb5(dg=1.0, density=0.4, mrl=2.0)['estimate'] - 1.6) < 1e-15
# True
