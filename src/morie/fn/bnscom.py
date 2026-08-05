# morie.fn -- function file (rootcoder007/morie)
"""Bound on the ATE under unknown compliance behaviour."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_compliance"]


def bound_compliance(y, D, Z):
    """Complier means, principal-strata shares, and the resulting ATE bound.

    Under instrument independence and monotonicity the population splits
    into compliers, always-takers and never-takers, whose shares are
    identified from the first stage.  The instrument reveals the treatment
    effect only for the compliers; for the other two strata one of the two
    potential outcomes is never observed at all, so the population ATE is
    a mixture of one identified piece and two entirely unidentified ones,
    and the interval simply admits the extremes for those.

    Derivation.  ``E[y D | Z = z] = sum over strata with d(z) = 1``, so
    differencing in ``z`` leaves only compliers:
    ``E[y(1) | c] = (E[y D | Z = 1] - E[y D | Z = 0]) / pi_c`` and
    ``E[y(0) | c] = (E[y (1 - D) | Z = 0] - E[y (1 - D) | Z = 1]) / pi_c``,
    with ``pi_c = P(D = 1 | Z = 1) - P(D = 1 | Z = 0)``.  Their difference
    is the Wald ratio.  The ATE bound is then
    ``pi_c LATE + (1 - pi_c) [y_0 - y_1, y_1 - y_0]``.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment, coded 0/1.
    Z : array-like
        Binary instrument, coded 0/1.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``late``,
        ``pi_c``, ``pi_a``, ``pi_n``, ``e1c``, ``e0c``, ``n``.

    References
    ----------
    Imbens, G. W. & Rubin, D. B. (1997).  Estimating outcome
    distributions for compliers in instrumental variables models.
    Review of Economic Studies 64(4), 555-574.  doi:10.2307/2971731.
    Angrist, J. D., Imbens, G. W. & Rubin, D. B. (1996).  Identification
    of causal effects using instrumental variables.  Journal of the
    American Statistical Association 91(434), 444-455.
    doi:10.1080/01621459.1996.10476902.
    """
    yv = C.vec(y)
    dv = C.vec(D)
    zv = C.vec(Z)
    n = len(yv)
    if n == 0:
        raise ValueError("bound_compliance: y is empty")
    if len(dv) != n or len(zv) != n:
        raise ValueError("bound_compliance: y, D and Z must have the same length")
    for v in list(dv) + list(zv):
        if v != 0.0 and v != 1.0:
            raise ValueError("bound_compliance: D and Z must be coded 0/1")
    acc = {0: [0, 0.0, 0.0, 0.0], 1: [0, 0.0, 0.0, 0.0]}
    for i in range(n):
        z = int(zv[i])
        acc[z][0] += 1
        acc[z][1] += dv[i]
        acc[z][2] += yv[i] * dv[i]
        acc[z][3] += yv[i] * (1.0 - dv[i])
    if acc[0][0] == 0 or acc[1][0] == 0:
        raise ValueError("bound_compliance: the instrument takes only one value")
    pd1 = acc[1][1] / acc[1][0]
    pd0 = acc[0][1] / acc[0][0]
    pi_c = pd1 - pd0
    if pi_c <= 0.0:
        raise ValueError("bound_compliance: non-positive first stage; monotonicity fails")
    pi_a = pd0
    pi_n = 1.0 - pd1
    e1c = (acc[1][2] / acc[1][0] - acc[0][2] / acc[0][0]) / pi_c
    e0c = (acc[0][3] / acc[0][0] - acc[1][3] / acc[1][0]) / pi_c
    late = e1c - e0c
    y0 = y1 = yv[0]
    for v in yv:
        if v < y0:
            y0 = v
        if v > y1:
            y1 = v
    rest = 1.0 - pi_c
    lo = pi_c * late + rest * (y0 - y1)
    hi = pi_c * late + rest * (y1 - y0)
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "late": late,
        "pi_c": pi_c, "pi_a": pi_a, "pi_n": pi_n,
        "e1c": e1c, "e0c": e0c, "n": n,
        "method": "Bound under unknown compliance"})


def cheatsheet():
    return "bnscom: Bound under unknown compliance"
