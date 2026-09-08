# morie.fn -- function file (rootcoder007/morie)
"""LATE bound when monotonicity may be violated."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_no_monotonicity"]


def bound_no_monotonicity(y, D, Z):
    """Bound the complier effect when defiers are not ruled out.

    The first stage identifies only the NET complier share,
    ``pi_c - pi_d``; monotonicity is what turns that into ``pi_c``.
    Without it the Wald ratio is a difference of two effects, and the
    data bound the defier share only by
    ``pi_c <= min(P(D = 1 | Z = 1), P(D = 0 | Z = 0))``.  The interval
    widens with the admissible defier share, so the union over admissible
    ``pi_c`` is attained at that maximum, and collapses to the Wald ratio
    exactly when the maximum equals the net share -- that is, when the
    data leave no room for defiers.

    Derivation: ``ITT_y = pi_c E[D | c] - pi_d E[D | d]`` with
    ``|E[D | d]| <= y_1 - y_0``, so with ``pi_d = pi_c - ITT_D``,
    ``LATE in [(ITT_y - pi_d R) / pi_c, (ITT_y + pi_d R) / pi_c]``
    at ``pi_c = pi_c^max``, ``R = y_1 - y_0``.

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
        ``lower``, ``upper``, ``width``, ``estimate``, ``wald``,
        ``pi_net``, ``pi_c_max``, ``pi_d_max``, ``itt_y``, ``n``.

    References
    ----------
    de Chaisemartin, C. (2017).  Tolerating defiance? Local average
    treatment effects without monotonicity.  Quantitative Economics 8(2),
    367-396.  doi:10.3982/QE601 -- the stub's attribution, for the
    problem; the paper was not accessible, so the interval above is the
    elementary mixture bound, derived in the docstring rather than taken
    from it.  His result is sharper: it identifies the effect on a
    "comvivor" subpopulation under a compliers-defiers condition.
    """
    yv = C.vec(y)
    dv = C.vec(D)
    zv = C.vec(Z)
    n = len(yv)
    if n == 0:
        raise ValueError("bound_no_monotonicity: y is empty")
    if len(dv) != n or len(zv) != n:
        raise ValueError("bound_no_monotonicity: y, D and Z must have the same length")
    for v in list(dv) + list(zv):
        if v != 0.0 and v != 1.0:
            raise ValueError("bound_no_monotonicity: D and Z must be coded 0/1")
    n1 = sum(1 for v in zv if v == 1.0)
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        raise ValueError("bound_no_monotonicity: the instrument takes only one value")
    sy1 = sum(yv[i] for i in range(n) if zv[i] == 1.0) / n1
    sy0 = sum(yv[i] for i in range(n) if zv[i] == 0.0) / n0
    pd1 = sum(dv[i] for i in range(n) if zv[i] == 1.0) / n1
    pd0 = sum(dv[i] for i in range(n) if zv[i] == 0.0) / n0
    itt_y = sy1 - sy0
    net = pd1 - pd0
    if net <= 0.0:
        raise ValueError("bound_no_monotonicity: non-positive net first stage")
    pc_max = pd1 if pd1 < (1.0 - pd0) else (1.0 - pd0)
    pd_max = pc_max - net
    y0 = y1 = yv[0]
    for v in yv:
        if v < y0:
            y0 = v
        if v > y1:
            y1 = v
    rng = y1 - y0
    lo = (itt_y - pd_max * rng) / pc_max
    hi = (itt_y + pd_max * rng) / pc_max
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "wald": itt_y / net,
        "pi_net": net, "pi_c_max": pc_max, "pi_d_max": pd_max,
        "itt_y": itt_y, "n": n,
        "method": "Bound when monotonicity violated"})


def cheatsheet():
    return "bndnmt: LATE bound allowing defiers"
