# morie.fn -- function file (rootcoder007/morie)
"""Block coordinate descent on a convex quadratic."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["block_coordinate_descent"]


def block_coordinate_descent(Q, b, blocks, x0=None, n_iter=20):
    """Cyclic block coordinate descent, exact minimisation per block.

    Tseng result is that cyclic block descent converges on a function
    that is smooth plus a separable non-smooth part, and that
    separability is what saves it -- descent on one block cannot be
    blocked by a kink in another.  For the convex quadratic implemented
    here each block subproblem has a closed form, so the sweep is exact
    rather than a line search, and the objective decreases monotonically
    by construction.

    Determinism: a fixed sweep count and a fixed block order, with no
    tolerance test.  The callable-objective form of Tseng theorem is
    not representable across the Python and R arms, so the quadratic
    case -- where the exact block minimiser exists in closed form -- is
    what is provided.

    Formula: minimise ``f(x) = 0.5 x' Q x - b' x``; the exact minimiser
    over block ``B`` with the rest held fixed is
    ``x_B = Q_BB^{-1} (b_B - Q_{B,B^c} x_{B^c})``.

    Parameters
    ----------
    Q : array-like, shape (p, p)
        Symmetric positive definite Hessian.
    b : array-like, shape (p,)
        Linear term.
    blocks : list of list of int
        Zero-based coordinate indices, one list per block.
    x0 : array-like, optional
        Starting point; zeros by default.
    n_iter : int, default 20
        Number of full sweeps over all blocks.

    Returns
    -------
    RichResult
        ``estimate`` (objective at the end), ``x``, ``obj_trace``,
        ``n_iter``.

    References
    ----------
    Tseng, P. (2001).  Convergence of a block coordinate descent method
    for nondifferentiable minimization.  Journal of Optimization Theory
    and Applications 109:475-494.
    """
    Qm = C.mat(Q)
    bv = C.vec(b)
    p = len(bv)
    x = list(C.vec(x0)) if x0 is not None else [0.0] * p
    blocks = [[int(j) for j in blk] for blk in blocks]
    def obj(v):
        return 0.5 * sum(v[i] * Qm[i][j] * v[j] for i in range(p) for j in range(p)) - C.dot(bv, v)
    trace = [obj(x)]
    for _ in range(int(n_iter)):
        for blk in blocks:
            rest = [j for j in range(p) if j not in blk]
            Qbb = [[Qm[i][j] for j in blk] for i in blk]
            rhs = [bv[i] - sum(Qm[i][j] * x[j] for j in rest) for i in blk]
            sol = C.solvev(Qbb, rhs)
            for t, i in enumerate(blk):
                x[i] = sol[t]
        trace.append(obj(x))
    return RichResult(payload={
        "estimate": trace[-1], "x": x, "obj_trace": trace, "n_iter": int(n_iter),
        "method": "Block coordinate descent, exact quadratic blocks"})


def cheatsheet():
    return "bcdblk: Block coordinate descent on a convex quadratic."
