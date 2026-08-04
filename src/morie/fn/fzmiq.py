# morie.fn -- function file (rootcoder007/morie)
"""Moment bound for the H-decomposition projections."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["hdecmom", "fauzi_moment_ineq_ustat"]


def hdecmom(n, k, q, rhomom=1.0, c=1.0):
    r"""Moment bound for the H-decomposition projections.

    Eq. (3.12): for :math:`q \ge 2`, if
    :math:`E|\nu(X_1,\dots,X_r)|^q < \infty` there is a constant ``C``
    depending on :math:`\nu` and ``F`` but NOT on ``n`` with

    .. math:: E|A_k|^q \le C n^{qk/2}\,
              E|\rho_k(X_{i_1},\dots,X_{i_k})|^q,

    where :math:`A_k = \sum_{C_{n,k}}\rho_k` is the ``k``-th projection of
    the H-decomposition (3.10)-(3.11).

    The exponent is the useful part. A naive count would put
    :math:`\binom nk \sim n^k` terms in :math:`A_k`, giving :math:`n^{qk}`;
    the martingale property (3.9),
    :math:`E[\rho_k|X_1,\dots,X_{k-1}] = 0`, halves the exponent to
    :math:`n^{qk/2}`. That square root is what makes the higher
    projections negligible and lets Lemma 3.1 stop at :math:`k = 3` with
    an :math:`o_L(n^{-1/2})` remainder.

    ``C`` is genuinely unspecified in the text -- it is a generic constant
    the book explicitly says "may change its meaning at different places".
    It defaults to 1 and the returned bound is labelled ``bound_over_c``
    for that reason; treating it as an absolute number would be a
    fabrication.

    Parameters
    ----------
    n : int
        Sample size.
    k : int
        Projection order, ``k >= 1``.
    q : float
        Moment order, ``q >= 2``.
    rhomom : float, default 1.0
        ``E|rho_k|^q``.
    c : float, default 1.0
        The unspecified constant of (3.12).

    Returns
    -------
    RichResult
        Keys ``bound_over_c``, ``bound``, ``exponent``, ``naive``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eqs. (3.9)-(3.13).
    """
    n = int(n)
    k = int(k)
    q = float(q)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if k < 1:
        raise ValueError(f"projection order must be at least 1, got {k}.")
    if q < 2:
        raise ValueError(f"(3.12) is stated for q >= 2, got {q}.")
    expo = q * k / 2.0
    scaled = float(n) ** expo * float(rhomom)
    return RichResult(
        payload={
            "bound_over_c": float(scaled),
            "bound": float(c) * scaled,
            "exponent": float(expo),
            "naive": float(q * k),
            "method": "H-decomposition moment bound (Eq. 3.12)",
        }
    )


fauzi_moment_ineq_ustat = hdecmom


def cheatsheet():
    return "fzmiq: E|A_k|^q <= C n^(qk/2): the martingale property halves the naive exponent (3.12)"


# CANONICAL TEST
# >>> r = hdecmom(n=100, k=2, q=2)
# >>> r['exponent'] == 2.0 and r['naive'] == 4.0
# True
