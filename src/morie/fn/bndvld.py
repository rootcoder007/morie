# morie.fn -- function file (rootcoder007/morie)
"""Validity check for a collection of bound assumptions."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_validity_check"]


def bound_validity_check(lower, upper, theta_0, H0=1.0):
    """Refutation and coverage check for a family of interval bounds.

    Each maintained assumption delivers its own interval for the same
    scalar parameter, so all of them together deliver the intersection.
    An empty intersection refutes the assumptions jointly -- this is the
    only sense in which a partial-identification analysis can be falsified,
    since a non-empty region is consistent with the data by construction.
    A non-empty intersection that excludes ``theta_0`` rejects the null.

    Formula: ``L = max_i lower_i``, ``U = min_i upper_i``; refuted iff
    ``L > U``; covers iff ``L <= theta_0 <= U``.

    Parameters
    ----------
    lower, upper : array-like
        Lower and upper end of each maintained bound, same length.
    theta_0 : float
        Parameter value under test.
    H0 : float, optional
        Non-zero to test coverage of ``theta_0`` (the default); zero to
        report the intersection only, leaving ``reject`` at 0.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``refuted``, ``covers``,
        ``reject``, ``n``.

    References
    ----------
    Manski, C. F. (2003).  Partial Identification of Probability
    Distributions.  Springer.  The refutation reading of an empty
    identification region is stated in Molinari, F. (2021), Handbook of
    Econometrics 7A (arXiv:2004.11751 p. 20): "If the bounds are empty, the
    mean independence assumption can be refuted".
    """
    lo = C.vec(lower)
    hi = C.vec(upper)
    if len(lo) == 0:
        raise ValueError("bound_validity_check: lower is empty")
    if len(hi) != len(lo):
        raise ValueError("bound_validity_check: lower and upper must have the same length")
    L = lo[0]
    U = hi[0]
    for v in lo:
        if v > L:
            L = v
    for v in hi:
        if v < U:
            U = v
    t0 = float(theta_0)
    refuted = 1.0 if L > U else 0.0
    covers = 1.0 if (L <= t0 and t0 <= U) else 0.0
    reject = 1.0 if (float(H0) != 0.0 and covers == 0.0) else 0.0
    return RichResult(payload={
        "lower": L, "upper": U, "width": U - L,
        "refuted": refuted, "covers": covers, "reject": reject,
        "n": len(lo), "method": "Validity check for bound assumptions"})


def cheatsheet():
    return "bndvld: Validity check for bound assumptions"
