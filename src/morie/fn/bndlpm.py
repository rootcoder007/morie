# morie.fn -- function file (rootcoder007/morie)
"""Balke-Pearl linear-programming bounds on the ATE with a binary instrument."""

from . import _tail1core as C

from ._richresult import RichResult
from .bndcvx import bound_convex_estimator

__all__ = ["bound_lp_method"]

_DZ = ((0, 0), (0, 1), (1, 0), (1, 1))
_YK = ((0, 0), (0, 1), (1, 0), (1, 1))


def bound_lp_method(y, D, Z, moment_eqs=None):
    """Sharp bounds on the ATE from the response-type linear program.

    With binary outcome, treatment and instrument every unit belongs to
    one of sixteen response types: how its treatment responds to the
    instrument, crossed with how its outcome responds to treatment.  The
    eight observed conditional probabilities are linear in the type
    shares, and the ATE is linear in them too, so the sharp bounds are
    the optimum of a linear program over the simplex.  Monotonicity is
    NOT imposed -- defiers keep their own share, and the program is what
    makes the bounds sharp rather than merely valid.

    Formula: ``min / max sum_jk q_jk [y_k(1) - y_k(0)]`` subject to
    ``sum_{j : d_j(z) = b} sum_{k : y_k(b) = a} q_jk = P(y = a, D = b | Z = z)``
    for all ``a, b, z`` and ``q >= 0``.

    Parameters
    ----------
    y : array-like
        Binary outcome, coded 0/1.
    D : array-like
        Binary treatment, coded 0/1.
    Z : array-like
        Binary instrument, coded 0/1.
    moment_eqs : array-like, shape (m, 17), optional
        Extra linear equality constraints on the sixteen type shares:
        each row is sixteen coefficients followed by its right-hand side.
        Default ``None``, no extra constraints.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``feasible``,
        ``n_constraints``, ``n``.

    References
    ----------
    Balke, A. & Pearl, J. (1997).  Bounds on treatment effects from
    studies with imperfect compliance.  Journal of the American
    Statistical Association 92(439), 1171-1176.
    doi:10.1080/01621459.1997.10474074.  The linear program is stated in
    the paper; the response-type parameterisation used here is the one
    described in Molinari, F. (2021), Handbook of Econometrics 7A
    (arXiv:2004.11751 p. 19 and note 10), which is the accessible copy.
    """
    yv = C.vec(y)
    dv = C.vec(D)
    zv = C.vec(Z)
    n = len(yv)
    if n == 0:
        raise ValueError("bound_lp_method: y is empty")
    if len(dv) != n or len(zv) != n:
        raise ValueError("bound_lp_method: y, D and Z must have the same length")
    for v in list(yv) + list(dv) + list(zv):
        if v != 0.0 and v != 1.0:
            raise ValueError("bound_lp_method: y, D and Z must be coded 0/1")
    nz = [0, 0]
    for v in zv:
        nz[int(v)] += 1
    if nz[0] == 0 or nz[1] == 0:
        raise ValueError("bound_lp_method: the instrument takes only one value")
    p = {}
    for z in (0, 1):
        for b in (0, 1):
            for a in (0, 1):
                cnt = 0
                for i in range(n):
                    if int(zv[i]) == z and int(dv[i]) == b and int(yv[i]) == a:
                        cnt += 1
                p[(a, b, z)] = cnt / float(nz[z])
    A = []
    bvec = []
    for z in (0, 1):
        for b in (0, 1):
            for a in (0, 1):
                row = [0.0] * 16
                for j in range(4):
                    if _DZ[j][z] != b:
                        continue
                    for k in range(4):
                        if _YK[k][b] == a:
                            row[j * 4 + k] = 1.0
                A.append(row)
                bvec.append(p[(a, b, z)])
    if moment_eqs is not None:
        M = C.mat(moment_eqs)
        for r in M:
            if len(r) != 17:
                raise ValueError("bound_lp_method: each extra constraint needs 17 entries")
            A.append([float(v) for v in r[:16]])
            bvec.append(float(r[16]))
    cvec = [0.0] * 16
    for j in range(4):
        for k in range(4):
            cvec[j * 4 + k] = float(_YK[k][1] - _YK[k][0])
    r = bound_convex_estimator(cvec, A_eq=A, b_eq=bvec,
                               bounds=[(0.0, 1.0)] * 16)
    lo = float(r["lower"])
    hi = float(r["upper"])
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi),
        "feasible": 1.0 if r["feasible"] else 0.0,
        "n_constraints": len(A), "n": n,
        "method": "Linear programming method for bounds"})


def cheatsheet():
    return "bndlpm: Balke-Pearl sharp ATE bounds by linear programming"
