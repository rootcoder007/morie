"""Sequential linear programming with a trust region on the l1 exact penalty.

Fletcher, R. and Sainz de la Maza, E. (1989). Nonlinear programming and nonsmooth optimization
by successive linear programming. Mathematical Programming 43, 235-256. Griffith, R. E. and
Stewart, R. A. (1961). A nonlinear programming technique for the optimization of continuous
processing systems. Management Science 7, 379-392.
"""

from ._qncore import num_grad
from ._qpcore import dot, simplex_standard, ssum
from ._richresult import RichResult

__all__ = ["sequential_linear_programming"]


def sequential_linear_programming(
    f, x0, grad=None, eq=(), ineq=(), eq_jac=None, ineq_jac=None, mu=1.0, delta=1.0, tol=1e-8, max_iter=500
):
    r"""Minimise f(x) subject to c_E(x) = 0, c_I(x) >= 0 by trust-region SLP.

    At x the model of the l1 penalty phi = f + mu (sum |c_E| + sum max(0, -c_I)) is
    linearised, m(p) = f + g'p + mu (sum |c_E + A_E p| + sum max(0, -(c_I + A_I p))), and
    minimised over |p_i| <= Delta as an elastic LP (always feasible and bounded), solved
    exactly by a two-phase tableau simplex with Bland's rule. SLP reaches
    vertex solutions in finitely many steps but converges only to first order at
    non-vertex optima (use SQP there). The step is
    accepted when rho = (phi(x) - phi(x + p)) / (m(0) - m(p)) > 0.1; Delta shrinks to
    ||p|| / 4 when rho < 0.25 and doubles when rho > 0.75 at the boundary. Stops when the
    predicted reduction or Delta falls below ``tol`` (``converged`` only for the former, and
    only at a feasible point: with too small an initial mu the l1 penalty can be unbounded
    below and the iterates run off, which is reported, not hidden); mu is multiplied by 10
    (up to 1e8) while the limit point is infeasible, so mu ends above the multipliers as
    exactness of the l1 penalty requires.

    Parameters
    ----------
    f : callable
    x0 : sequence of float
    grad : callable, optional
    eq, ineq : sequences of callables
    eq_jac, ineq_jac : callables, optional
    mu : float
        Initial penalty weight.
    delta : float
        Initial trust-region radius (infinity norm).
    tol : float
    max_iter : int

    Returns
    -------
    RichResult
        Keys: x, fun, violation, penalty, radius, n_iter, converged.

    References
    ----------
    Fletcher, R. and Sainz de la Maza, E. (1989). Mathematical Programming 43, 235-256.

    Examples
    --------
    >>> r = sequential_linear_programming(lambda x: (x[0] - 5) ** 2 + (x[1] - 5) ** 2, [0.0, 0.0], ineq=[lambda x: 1 - x[0], lambda x: 2 - x[1]])
    >>> [round(v, 8) for v in r["x"]]
    [1.0, 2.0]
    """
    eq, ineq = list(eq), list(ineq)
    gr = grad if grad is not None else (lambda z: num_grad(f, z))
    je = eq_jac if eq_jac is not None else (lambda z: [num_grad(c, z) for c in eq])
    ji = ineq_jac if ineq_jac is not None else (lambda z: [num_grad(c, z) for c in ineq])
    x = [float(v) for v in x0]
    n, me, mi = len(x), len(eq), len(ineq)
    mu, D = float(mu), float(delta)

    def cons(z):
        return [float(c(z)) for c in eq], [float(c(z)) for c in ineq]

    def viol(ce, ci):
        return ssum(abs(v) for v in ce) + ssum(max(0.0, -v) for v in ci)

    fx = float(f(x))
    ce, ci = cons(x)
    it, conv = 0, False
    while it < max_iter:
        g = [float(v) for v in gr(x)]
        AE, AI = (je(x) if me else []), (ji(x) if mi else [])
        nv = 2 * n + 2 * me + 2 * mi
        c = g + [0.0] * n + [mu] * (2 * me) + [mu] * mi + [0.0] * mi
        A, b = [], []
        for j in range(n):  # q_j + w_j = 2 Delta, p = q - Delta
            row = [0.0] * nv
            row[j], row[n + j] = 1.0, 1.0
            A.append(row)
            b.append(2 * D)
        for k in range(me):  # a'q - e+ + e- = -c + Delta sum a
            row = [0.0] * nv
            row[:n] = list(AE[k])
            row[2 * n + k], row[2 * n + me + k] = -1.0, 1.0
            A.append(row)
            b.append(-ce[k] + D * ssum(AE[k]))
        for k in range(mi):  # a'q + t - sigma = -c + Delta sum a
            row = [0.0] * nv
            row[:n] = list(AI[k])
            row[2 * n + 2 * me + k], row[2 * n + 2 * me + mi + k] = 1.0, -1.0
            A.append(row)
            b.append(-ci[k] + D * ssum(AI[k]))
        sol, _ = simplex_standard(c, A, b)
        p = [min(max(sol[j] - D, -D), D) for j in range(n)]
        mp = (
            fx
            + dot(g, p)
            + mu
            * (
                ssum(abs(ce[k] + dot(AE[k], p)) for k in range(me))
                + ssum(max(0.0, -(ci[k] + dot(AI[k], p))) for k in range(mi))
            )
        )
        phi = fx + mu * viol(ce, ci)
        pred = phi - mp
        if pred <= tol * max(1.0, abs(phi)) or tol >= D:
            if viol(ce, ci) > tol and mu < 1e8:
                mu *= 10.0
                D = max(D, float(delta))
                it += 1
                continue
            conv = pred <= tol * max(1.0, abs(phi)) and viol(ce, ci) <= tol
            break
        xn = [xi + pi for xi, pi in zip(x, p)]
        fn = float(f(xn))
        cen, cin = cons(xn)
        rho = (phi - (fn + mu * viol(cen, cin))) / pred
        pn = max(abs(v) for v in p)
        if rho > 0.1:
            x, fx, ce, ci = xn, fn, cen, cin
        if rho < 0.25:
            D = 0.25 * pn
        elif rho > 0.75 and pn >= 0.99 * D:
            D = 2.0 * D
        it += 1
    return RichResult(
        title="Sequential linear programming",
        summary_lines=[("f", fx), ("iterations", it)],
        payload={
            "x": x,
            "fun": fx,
            "violation": viol(ce, ci),
            "penalty": mu,
            "radius": D,
            "n_iter": it,
            "converged": conv,
        },
    )


def cheatsheet():
    return "slpmin: trust-region SLP on the l1 penalty (Fletcher and Sainz de la Maza 1989)"
