"""Line-search SQP with damped BFGS and an l1 merit function.

Nocedal, J. and Wright, S. J. (2006). Numerical Optimization, 2nd ed., Algorithm 18.3 and
Procedure 18.2 (Powell's damped BFGS update). Han, S.-P. (1977). A globally convergent method
for nonlinear programming. Journal of Optimization Theory and Applications 22, 297-309.
"""

from ._qncore import num_grad
from ._qpcore import dot, goldfarb_idnani, matvec, ssum
from ._richresult import RichResult

__all__ = ["sequential_quadratic_programming"]


def _jac(fs, x):
    return [num_grad(c, x) for c in fs]


def sequential_quadratic_programming(
    f, x0, grad=None, eq=(), ineq=(), eq_jac=None, ineq_jac=None, tol=1e-8, max_iter=200
):
    r"""Minimise f(x) subject to c_E(x) = 0 and c_I(x) >= 0 by line-search SQP.

    Each iteration solves the QP min 1/2 p'Bp + g'p s.t. A_E p + c_E = 0,
    A_I p + c_I >= 0 exactly (Goldfarb-Idnani dual active set), raises the penalty to
    mu = max(mu, 1.1 max |lambda|), and backtracks (halving from 1) until the l1 merit
    phi = f + mu (sum |c_E| + sum max(0, -c_I)) falls by at least 1e-4 a D, with
    D = g'p - mu (sum |c_E| + sum max(0, -c_I)) its directional derivative. B is updated
    by Powell's damped BFGS on s and y = grad L(x+, lambda) - grad L(x, lambda),
    L = f - lambda'c. Stops when the Lagrangian gradient and the constraint violation are
    both below ``tol``.

    Parameters
    ----------
    f : callable
    x0 : sequence of float
    grad : callable, optional
        Gradient of f (central differences when omitted).
    eq, ineq : sequences of callables
        Equality constraints c(x) = 0 and inequality constraints c(x) >= 0.
    eq_jac, ineq_jac : callables, optional
        Each returns the list of constraint gradients at x.
    tol : float
    max_iter : int

    Returns
    -------
    RichResult
        Keys: x, fun, multipliers_eq, multipliers_ineq, kkt_residual, violation,
        n_iter, converged.

    References
    ----------
    Nocedal, J. and Wright, S. J. (2006). Numerical Optimization, Algorithm 18.3.
    Han, S.-P. (1977). Journal of Optimization Theory and Applications 22, 297-309.

    Examples
    --------
    >>> r = sequential_quadratic_programming(lambda x: x[0] ** 2 + x[1] ** 2, [2.0, 0.0], eq=[lambda x: x[0] + x[1] - 1])
    >>> [round(v, 8) for v in r["x"]]
    [0.5, 0.5]
    """
    eq, ineq = list(eq), list(ineq)
    gr = grad if grad is not None else (lambda z: num_grad(f, z))
    je = eq_jac if eq_jac is not None else (lambda z: _jac(eq, z))
    ji = ineq_jac if ineq_jac is not None else (lambda z: _jac(ineq, z))
    x = [float(v) for v in x0]
    n, me, mi = len(x), len(eq), len(ineq)

    def cons(z):
        return [float(c(z)) for c in eq], [float(c(z)) for c in ineq]

    def viol(ce, ci):
        return ssum(abs(v) for v in ce) + ssum(max(0.0, -v) for v in ci)

    def lag_grad(g, AE, AI, le, li):
        return [
            g[j] - ssum(le[k] * AE[k][j] for k in range(me)) - ssum(li[k] * AI[k][j] for k in range(mi))
            for j in range(n)
        ]

    B = [[float(i == j) for j in range(n)] for i in range(n)]
    mu = 0.0
    le, li = [0.0] * me, [0.0] * mi
    fx, g = float(f(x)), [float(v) for v in gr(x)]
    ce, ci = cons(x)
    AE, AI = (je(x) if me else []), (ji(x) if mi else [])
    it, conv = 0, False
    kkt = max(abs(v) for v in lag_grad(g, AE, AI, le, li))
    while it < max_iter:
        C = [list(r) for r in AE] + [list(r) for r in AI]
        bq = [-v for v in ce] + [-v for v in ci]
        p, u, _ = goldfarb_idnani(B, g, C, bq, me)
        le, li = u[:me], u[me:]
        kkt = max(abs(v) for v in lag_grad(g, AE, AI, le, li))
        v0 = viol(ce, ci)
        if kkt <= tol and v0 <= tol:
            conv = True
            break
        if max(abs(v) for v in p) <= 1e-15 * max(1.0, max(abs(v) for v in x)):
            conv = kkt <= 1e3 * tol and v0 <= 1e3 * tol
            break
        mu = max(mu, 1.1 * max((abs(v) for v in u), default=0.0), 1e-8)
        phi0 = fx + mu * v0
        D = dot(g, p) - mu * v0
        a = 1.0
        while True:
            xn = [xi + a * pi for xi, pi in zip(x, p)]
            fn = float(f(xn))
            cen, cin = cons(xn)
            if fn + mu * viol(cen, cin) <= phi0 + 1e-4 * a * D or a < 1e-10:
                break
            a *= 0.5
        gn = [float(v) for v in gr(xn)]
        AEn, AIn = (je(xn) if me else []), (ji(xn) if mi else [])
        s = [u_ - v_ for u_, v_ in zip(xn, x)]
        y = [u_ - v_ for u_, v_ in zip(lag_grad(gn, AEn, AIn, le, li), lag_grad(g, AE, AI, le, li))]
        Bs = matvec(B, s)
        sBs, sy = dot(s, Bs), dot(s, y)
        if sBs > 1e-300:
            th = 1.0 if sy >= 0.2 * sBs else 0.8 * sBs / (sBs - sy)
            r = [th * yi + (1 - th) * bi for yi, bi in zip(y, Bs)]
            sr = dot(s, r)
            B = [[B[i][j] - Bs[i] * Bs[j] / sBs + r[i] * r[j] / sr for j in range(n)] for i in range(n)]
        x, fx, g, ce, ci, AE, AI = xn, fn, gn, cen, cin, AEn, AIn
        it += 1
    return RichResult(
        title="Sequential quadratic programming",
        summary_lines=[("f", fx), ("iterations", it), ("KKT residual", kkt)],
        payload={
            "x": x,
            "fun": fx,
            "multipliers_eq": le,
            "multipliers_ineq": li,
            "kkt_residual": kkt,
            "violation": viol(ce, ci),
            "n_iter": it,
            "converged": conv,
        },
    )


def cheatsheet():
    return "sqpmin: line-search SQP, Goldfarb-Idnani QP subproblems, damped BFGS, l1 merit"
