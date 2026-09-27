"""Range a correlation can take given its correlations with a third variable."""

import math

from ._richresult import RichResult

__all__ = ["correlation_admissible_range"]


def correlation_admissible_range(r_ik, r_jk):
    r"""Bounds on :math:`r_{ij}` implied by :math:`r_{ik}` and :math:`r_{jk}`.

    A 3 x 3 correlation matrix is positive semi-definite only if
    (Hedderich, Sachs & Reynarowych 2023, p. 770)

    .. math::  r_{ik} r_{jk} - \sqrt{(1-r_{ik}^2)(1-r_{jk}^2)} \le r_{ij}
               \le r_{ik} r_{jk} + \sqrt{(1-r_{ik}^2)(1-r_{jk}^2)}.

    Parameters
    ----------
    r_ik, r_jk : float
        Correlations in [-1, 1].

    Returns
    -------
    RichResult
        ``lower``, ``upper``.

    References
    ----------
    Hedderich, J., Sachs, L. & Reynarowych, Z. (2023). Applied Statistics:
    Methods Using R. Springer, p. 770.
    """
    a, b = float(r_ik), float(r_jk)
    if not (-1 <= a <= 1 and -1 <= b <= 1):
        raise ValueError("correlations must lie in [-1, 1]")
    h = math.sqrt((1 - a * a) * (1 - b * b))
    lo, hi = a * b - h, a * b + h
    return RichResult(
        title="Admissible range of r_ij",
        summary_lines=[("lower", lo), ("upper", hi)],
        payload={"lower": lo, "upper": hi},
    )


def cheatsheet():
    return "corrng: r_ik r_jk -/+ sqrt((1 - r_ik^2)(1 - r_jk^2))"
