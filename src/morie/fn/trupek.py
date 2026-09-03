"""Trust-region minimisation (Conn, Gould & Toint 2000).

A trust region is the ball around the current point inside which a
quadratic model of the objective is believed. Each iteration solves the
model on that ball, compares the reduction the model PREDICTED against the
reduction actually achieved, and lets that ratio decide both whether to
take the step and what the radius should be next time. The bookkeeping is
Algorithm BTR (6.1.1); the interesting part is the subproblem, and the
book gives several ways to solve it, so all of them are here.

  cauchy    The Cauchy point: minimise the model along the steepest
            descent direction within the ball. The cheapest step that
            still guarantees convergence (Theorem 6.3.1), and the
            benchmark every better method has to beat.
  dogleg    Powell's dogleg: the path from the Cauchy point to the
            Newton point, cut where it leaves the ball. Needs a positive
            definite Hessian.
  steihaug  Steihaug-Toint truncated conjugate gradients (Algorithm
            7.5.1): run CG on the model and stop at the first iterate
            that either leaves the ball or meets a direction of
            non-positive curvature, taking the boundary point in both
            cases. Handles indefinite Hessians and never needs a
            factorisation.
  exact     The More-Sorensen characterisation (Section 7.3): the
            solution satisfies (H + lambda I)s = -g with lambda >= 0,
            H + lambda I positive semidefinite, and lambda(||s|| - Delta)
            = 0. Solved by bisection on lambda, which is safe and, unlike
            Newton on the secular equation, cannot be thrown by the hard
            case.

The radius update uses the book's constants: shrink on a bad ratio,
enlarge only on a very good ratio that also sits on the boundary. Nothing
here is stochastic, so a run reproduces exactly.

Reference
  Conn, A.R., Gould, N.I.M. & Toint, P.L. (2000) "Trust-Region Methods."
    MPS-SIAM Series on Optimization, SIAM, Philadelphia. Algorithm
    6.1.1 (BTR), Section 6.3 (the Cauchy point), Section 7.3 (the exact
    subproblem), Algorithm 7.5.1 (Steihaug-Toint).
  Powell, M.J.D. (1970) "A new algorithm for unconstrained
    optimization", in Nonlinear Programming, Academic Press, 31-65.
    The dogleg path.
  Steihaug, T. (1983) "The conjugate gradient method and trust regions
    in large scale optimization", SIAM Journal on Numerical Analysis
    20(3), 626-637, doi:10.1137/0720042
"""

import math

from ._richresult import RichResult

__all__ = ["trust_region", "trupek", "cheatsheet"]

_SUBS = ("steihaug", "cauchy", "dogleg", "exact")


def _dot(a, b):
    """Compensated (Neumaier) inner product.

    Neither language's sum() is a plain double loop, and they are not
    unfaithful in the same way: CPython 3.12 and later apply Neumaier
    compensation to a run of floats, while R accumulates in long double,
    80 bits on x86. Writing a naive loop in both does not help either --
    it just makes both arms equally wrong, and a trust-region method is
    chaotic in the last bit, because norms are compared against the
    radius and one bit decides which branch of the CG loop is taken. In
    this module the two arms parted company at the third iteration.

    So the arithmetic is written out rather than delegated. Every
    operation here is specified by IEEE 754, so both arms produce the
    same bits, and the result is more accurate than either default.
    """
    s = 0.0
    c = 0.0
    for x, y in zip(a, b):
        t = x * y
        u = s + t
        if abs(s) >= abs(t):
            c += (s - u) + t
        else:
            c += (t - u) + s
        s = u
    return s + c


def _csum(vals):
    """The same compensation over a plain sequence."""
    s = 0.0
    c = 0.0
    for t in vals:
        u = s + t
        if abs(s) >= abs(t):
            c += (s - u) + t
        else:
            c += (t - u) + s
        s = u
    return s + c


def _norm(a):
    # math.sqrt, not the exponent operator. R's ^ special-cases an
    # exponent of one half to sqrt(); Python's ** goes to libm pow().
    # The two agree almost always, and "almost" is enough to move a norm
    # by one bit, flip the comparison against the trust radius, and send
    # the two arms down different branches of the CG loop from there on.
    return math.sqrt(_dot(a, a))


def _matvec(H, v):
    return [_dot(H[i], v) for i in range(len(H))]


def _axpy(a, x, y):
    return [a * xi + yi for xi, yi in zip(x, y)]


def _model(g, H, s):
    """The quadratic model's value at s, relative to the current point."""
    return _dot(g, s) + 0.5 * _dot(s, _matvec(H, s))


def _boundary_step(z, d, delta):
    """The positive tau with ||z + tau d|| = delta. Solving the quadratic
    directly loses precision when the two roots differ wildly, so take the
    stable root and recover the other from the product."""
    dd = _dot(d, d)
    zd = _dot(z, d)
    zz = _dot(z, z)
    disc = zd * zd - dd * (zz - delta * delta)
    if disc < 0.0:
        disc = 0.0
    sq = math.sqrt(disc)
    if zd >= 0.0:
        tau = (-zd - sq) / dd if dd > 0 else 0.0
        tau2 = (zz - delta * delta) / (-zd - sq) if (zd + sq) != 0 else tau
        tau = max(tau, tau2)
    else:
        tau = (-zd + sq) / dd if dd > 0 else 0.0
    return tau


def _cauchy(g, H, delta):
    gn = _norm(g)
    if gn == 0.0:
        return [0.0] * len(g)
    curv = _dot(g, _matvec(H, g))
    if curv <= 0.0:
        t = delta / gn
    else:
        t = min(gn * gn / curv, delta / gn)
    return [-t * gi for gi in g]


def _chol(H):
    """Cholesky, or None when H is not positive definite. Used both to
    test definiteness and to solve, so there is one code path."""
    n = len(H)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = H[i][j] - _dot(L[i][:j], L[j][:j])
            if i == j:
                if s <= 0.0:
                    return None
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _chol_solve(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - _dot(L[i][:i], y[:i])) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - _dot([L[k][i] for k in range(i + 1, n)],
                            x[i + 1:])) / L[i][i]
    return x


def _dogleg(g, H, delta):
    L = _chol(H)
    if L is None:
        return _cauchy(g, H, delta)     # no Newton point to aim at
    pb = [-v for v in _chol_solve(L, g)]
    if _norm(pb) <= delta:
        return pb
    curv = _dot(g, _matvec(H, g))
    if curv <= 0.0:
        return _cauchy(g, H, delta)
    pu = [-(_dot(g, g) / curv) * gi for gi in g]
    if _norm(pu) >= delta:
        gn = _norm(g)
        return [-(delta / gn) * gi for gi in g]
    d = [b - u for b, u in zip(pb, pu)]
    tau = _boundary_step(pu, d, delta)
    return _axpy(tau, d, pu)


def _steihaug(g, H, delta, tol, maxit):
    n = len(g)
    z = [0.0] * n
    r = list(g)
    d = [-v for v in g]
    gn = _norm(g)
    if gn == 0.0:
        return z, "zero gradient"
    stop = min(0.5, math.sqrt(gn)) * gn      # Algorithm 7.5.1's residual test
    stop = max(stop, tol)
    for _ in range(maxit if maxit else 2 * n + 1):
        Hd = _matvec(H, d)
        dHd = _dot(d, Hd)
        if dHd <= 0.0:
            tau = _boundary_step(z, d, delta)
            return _axpy(tau, d, z), "negative curvature, stopped on the boundary"
        alpha = _dot(r, r) / dHd
        z_next = _axpy(alpha, d, z)
        if _norm(z_next) >= delta:
            tau = _boundary_step(z, d, delta)
            return _axpy(tau, d, z), "left the region, stopped on the boundary"
        r_next = _axpy(alpha, Hd, r)
        if _norm(r_next) < stop:
            return z_next, "interior, residual below tolerance"
        beta = _dot(r_next, r_next) / _dot(r, r)
        d = _axpy(beta, d, [-v for v in r_next])
        z, r = z_next, r_next
    return z, "iteration limit"


def _exact(g, H, delta, tol, maxit):
    """More-Sorensen by bisection on lambda.

    Newton on the secular equation is faster and is what a production
    code uses, but it has to special-case the hard case, where the
    Newton step is undefined. Bisection has no hard case: the shifted
    matrix is positive definite for every lambda past the lower bound,
    and ||s(lambda)|| falls monotonically, so bracketing always works.
    """
    n = len(g)
    L = _chol(H)
    if L is not None:
        s = [-v for v in _chol_solve(L, g)]
        if _norm(s) <= delta:
            return s, 0.0, "interior, Hessian positive definite"
    # A lambda that certainly makes H + lambda I positive definite:
    # Gershgorin gives one, and it is an upper bound on what is needed.
    lo = 0.0
    hi = 1.0
    for i in range(n):
        row = -H[i][i] + _csum([abs(H[i][j]) for j in range(n) if j != i])
        if row > hi:
            hi = row
    hi = max(hi, _norm(g) / delta + 1.0)
    while True:
        Ls = _chol([[H[i][j] + (hi if i == j else 0.0) for j in range(n)]
                    for i in range(n)])
        if Ls is not None and _norm(_chol_solve(Ls, g)) <= delta:
            break
        hi *= 2.0
        if hi > 1e300:
            break
    s = [0.0] * n
    lam = hi
    for _ in range(maxit if maxit else 200):
        lam = 0.5 * (lo + hi)
        Ls = _chol([[H[i][j] + (lam if i == j else 0.0) for j in range(n)]
                    for i in range(n)])
        if Ls is None:
            lo = lam
            continue
        s = [-v for v in _chol_solve(Ls, g)]
        ns = _norm(s)
        if abs(ns - delta) <= tol * delta:
            break
        if ns > delta:
            lo = lam
        else:
            hi = lam
    return s, lam, "on the boundary, shifted by lambda"


def _solve_sub(g, H, delta, sub, tol, maxit):
    if sub == "cauchy":
        return _cauchy(g, H, delta), "Cauchy point"
    if sub == "dogleg":
        return _dogleg(g, H, delta), "dogleg path"
    if sub == "exact":
        s, lam, why = _exact(g, H, delta, tol, maxit)
        return s, why
    return _steihaug(g, H, delta, tol, maxit)


def trust_region(f, grad_f, hess_f, x0, delta=1.0, delta_max=None,
                 subproblem="steihaug", eta1=0.01, eta2=0.9, gamma1=0.5,
                 gamma3=2.0, max_iter=200, gtol=1e-10, dtol=1e-14,
                 sub_tol=1e-12, sub_maxit=0):
    """Minimise f from x0 by the basic trust-region algorithm.

    Parameters
    ----------
    f, grad_f, hess_f : callables
        Objective, gradient and Hessian at a point given as a list.
    x0 : sequence
        Starting point.
    delta : float
        Initial radius.
    delta_max : float, optional
        Cap on the radius. Defaults to 1e3 times the initial radius.
    subproblem : str
        steihaug, cauchy, dogleg or exact.
    eta1, eta2 : float
        Accept the step when the ratio reaches eta1; enlarge the radius
        only when it reaches eta2.
    gamma1, gamma3 : float
        Shrink and enlarge factors.
    gtol : float
        Stop when the gradient norm falls below this.
    dtol : float
        Stop when the radius collapses below this. Near the solution the
        predicted reduction drops under the rounding noise in f, the
        ratio stops meaning anything, and every step gets rejected --
        without this the loop just shrinks to zero and burns max_iter.

    Returns
    -------
    RichResult
        x, fval, gnorm, delta, iterations, accepted, rejected, converged,
        history of (f, gnorm, delta, rho), subproblem, method.
    """
    if subproblem not in _SUBS:
        raise ValueError("trupek: subproblem = %r; expected one of %s"
                         % (subproblem, ", ".join(_SUBS)))
    x = [float(v) for v in x0]
    if delta_max is None:
        delta_max = 1e3 * delta
    fx = float(f(x))
    acc = 0
    rej = 0
    hist = []
    conv = False
    last_why = "not started"
    why = "iteration limit"
    k = 0
    for k in range(1, int(max_iter) + 1):
        g = [float(v) for v in grad_f(x)]
        gn = _norm(g)
        if gn <= gtol:
            conv = True
            why = "gradient below gtol"
            hist.append((fx, gn, delta, 0.0))
            break
        if delta <= dtol:
            why = "radius collapsed below dtol"
            hist.append((fx, gn, delta, 0.0))
            break
        H = [[float(v) for v in row] for row in hess_f(x)]
        s, last_why = _solve_sub(g, H, delta, subproblem, sub_tol, sub_maxit)
        pred = -_model(g, H, s)
        if pred <= 0.0:
            # The model promises no descent, so there is nothing to test
            # the step against; shrink and try a smaller region.
            delta *= gamma1
            rej += 1
            hist.append((fx, gn, delta, 0.0))
            continue
        xt = [xi + si for xi, si in zip(x, s)]
        ft = float(f(xt))
        rho = (fx - ft) / pred
        hist.append((fx, gn, delta, rho))
        if rho >= eta1:
            x, fx = xt, ft
            acc += 1
        else:
            rej += 1
        sn = _norm(s)
        if rho < eta1:
            delta = gamma1 * delta
        elif rho >= eta2 and sn >= (1.0 - 1e-12) * delta:
            delta = min(gamma3 * delta, delta_max)
    g = [float(v) for v in grad_f(x)]
    return RichResult(payload={
        "exit_reason": why,
        "x": x,
        "fval": fx,
        "gnorm": _norm(g),
        "delta": delta,
        "iterations": k,
        "accepted": acc,
        "rejected": rej,
        "converged": conv,
        "history": hist,
        "subproblem": subproblem,
        "subproblem_exit": last_why,
        "method": "basic trust region (Conn, Gould & Toint 2000, Algorithm "
                  "6.1.1) with the %s subproblem" % subproblem,
    })


trupek = trust_region


def cheatsheet():
    return ("trupek: trust-region minimisation. subproblem = steihaug "
            "(truncated CG, the default) | cauchy | dogleg | exact "
            "(More-Sorensen by bisection).")


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
trustregion = trust_region
