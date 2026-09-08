# morie.fn -- function file (rootcoder007/morie)
"""The b_4 coefficient of the mean-residual-life variance (Eq. 4.28)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["mrlb4", "fauzi_b4_coefficient_mrl"]


def mrlb4(surv, cumsurv, mrl=None):
    r"""The b_4 coefficient of the mean-residual-life variance (Eq. 4.28).

    Eq. (4.28), first half:

    .. math:: b_4(t) = 2\bar S_X(t) - S_X(t)m_X^2(t),

    with :math:`\bar S_X` the cumulative survival
    :math:`\int_t^\infty S_X(u)du` and
    :math:`m_X(t) = \bar S_X(t)/S_X(t)` the mean residual life.

    It is the numerator of the leading :math:`1/n` term of
    :math:`\mathrm{Var}[m_{X,i}(t)]` in (4.27), divided by
    :math:`S_X^2(t)`.

    Substituting :math:`m_X = \bar S_X/S_X` gives
    :math:`b_4 = 2\bar S_X - \bar S_X^2/S_X`, so the whole variance term
    is :math:`\bar S_X(2S_X - \bar S_X)/(nS_X^3)`. That form makes the
    failure mode visible: as :math:`t` moves into the tail
    :math:`S_X\to0` cubed in the denominator, and the mean residual life
    becomes unestimable long before the survival function does.

    Supply ``mrl`` or let it be computed as ``cumsurv / surv``.

    Parameters
    ----------
    surv : float
        ``S_X(t)``, strictly positive.
    cumsurv : float
        ``bar S_X(t)``.
    mrl : float, optional
        ``m_X(t)``; defaults to ``cumsurv / surv``.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``mrl``, ``varterm``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eqs. (4.27)-(4.28).
    """
    s = float(surv)
    if s <= 0:
        raise ValueError(f"S_X(t) must be positive, got {s}.")
    m = float(cumsurv) / s if mrl is None else float(mrl)
    val = 2.0 * float(cumsurv) - s * m * m
    return RichResult(
        payload={
            "estimate": float(val),
            "mrl": float(m),
            "varterm": float(val / (s * s)),
            "method": "b_4 coefficient of the MRL variance (Eq. 4.28)",
        }
    )


fauzi_b4_coefficient_mrl = mrlb4


def cheatsheet():
    return "fzb4t: b_4 = 2 Sbar - S m^2; the MRL variance carries S^3 in the denominator (4.28)"


# CANONICAL TEST
# >>> r = mrlb4(surv=0.5, cumsurv=1.0)
# >>> abs(r['mrl'] - 2.0) < 1e-15 and abs(r['estimate'] - 0.0) < 1e-15
# True
