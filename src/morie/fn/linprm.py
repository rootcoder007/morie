# morie.fn -- function file (rootcoder007/morie)
r"""Linear programming: the vertex method and the interior one.

Two genuinely different algorithms solve the same problem, and the
difference is not speed alone.

**Simplex** (:mod:`morie.fn.clpopt`) walks the vertices of the
feasible polyhedron. It terminates at a *vertex*, so it returns a
basic solution -- exactly the right thing when the answer should be a
corner, and integral data often gives an integral answer. Its
worst-case running time is exponential; in practice it is very fast.

**Interior point** never touches the boundary until the end. It
follows the central path :math:`x_i s_i = \mu` toward
:math:`\mu \to 0`, taking Newton steps on the perturbed optimality
conditions

.. math:: A x = b, \quad A'y + s = c, \quad XSe = \mu e,
          \quad x, s > 0.

Karmarkar's result was that this can be done in polynomial time;
Mehrotra's predictor-corrector is what made it practical -- an affine
"predictor" step estimates how much progress is available, its
success sets the centring parameter :math:`\sigma =
(\mu_{\text{aff}}/\mu)^3`, and a "corrector" step then absorbs the
second-order error. Both steps reuse one factorisation of
:math:`A D A'`, so the corrector is nearly free.

**Where they disagree, and why that is informative.** When the optimal
face is not a single point, simplex returns a vertex of it and
interior point converges to its *analytic centre* -- a strictly
interior point of the face. Both are optimal; they are different
points, and the objective values agree to solver tolerance. The
anchor exhibits exactly this on a program with a flat optimal edge,
rather than pretending the two methods are interchangeable.

**The duality gap is the stopping rule and the certificate.** At
optimality :math:`c'x - b'y = x's = 0`. The gap is reported, so a
"solution" that has not actually converged cannot be mistaken for
one.

References
----------
Dantzig, G. B. (1963) *Linear Programming and Extensions*, Princeton
University Press, doi:10.1515/9781400884179, for the simplex method;
see :mod:`morie.fn.clpopt`.

Karmarkar, N. (1984) "A new polynomial-time algorithm for linear
programming", *Combinatorica* 4(4), 373-395, doi:10.1007/BF02579150.
That linear programming admits a polynomial-time interior algorithm,
and the central-path idea the modern methods descend from.

Mehrotra, S. (1992) "On the implementation of a primal-dual interior
point method", *SIAM Journal on Optimization* 2(4), 575-601,
doi:10.1137/0802028. The predictor-corrector scheme reproduced above:
the affine-scaling predictor, the adaptive centring parameter
:math:`\sigma = (\mu_{\text{aff}}/\mu)^3`, the second-order corrector
sharing one factorisation, and the starting-point heuristic.
"""

import math

from ._richresult import RichResult
from .clpopt import linprog as _simplex_solve
from .clpopt import standard_form

__all__ = ["interior_point", "solve_lp", "METHODS", "linear_program"]

METHODS = ("simplex", "interior_point", "auto")


def _cholesky(M):
    n = len(M)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = M[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 1e-14:
                    s = 1e-14   # the normal equations go ill-
                    # conditioned as the iterates approach the
                    # boundary; this is the standard regularisation
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _chol_solve(L, b):
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k]
                           for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[k][i] * x[k]
                           for k in range(i + 1, n))) / L[i][i]
    return x


def _ada(A, d):
    m, n = len(A), len(A[0])
    return [[sum(A[i][k] * d[k] * A[j][k] for k in range(n))
             for j in range(m)] for i in range(m)]


def interior_point(c, A, b, tol=1e-10, max_iter=200):
    r"""Mehrotra's predictor-corrector on :math:`Ax = b,\ x \ge 0`."""
    cv = [float(v) for v in c]
    M = [[float(v) for v in row] for row in A]
    bb = [float(v) for v in b]
    m, n = len(M), len(cv)
    if m == 0 or any(len(r) != n for r in M) or len(bb) != m:
        raise ValueError("linprm: A must be %d by %d with a "
                         "right-hand side of length %d"
                         % (m, n, m))
    x = [1.0] * n
    s = [1.0] * n
    y = [0.0] * m
    hist = []
    for it in range(int(max_iter)):
        rp = [bb[i] - sum(M[i][j] * x[j] for j in range(n))
              for i in range(m)]
        rd = [cv[j] - sum(M[i][j] * y[i] for i in range(m)) - s[j]
              for j in range(n)]
        mu = sum(x[j] * s[j] for j in range(n)) / n
        gap = sum(x[j] * s[j] for j in range(n))
        pr = math.sqrt(sum(v * v for v in rp))
        dr = math.sqrt(sum(v * v for v in rd))
        hist.append(mu)
        if gap < tol and pr < tol and dr < tol:
            return {"x": x, "y": y, "s": s, "iterations": it,
                    "gap": gap, "primal_residual": pr,
                    "dual_residual": dr, "converged": True,
                    "mu_history": hist}
        d = [x[j] / s[j] for j in range(n)]
        L = _cholesky(_ada(M, d))

        def step(r3):
            # dy from A D A' dy = rp - A S^-1 r3 + A D rd
            t = [r3[j] / s[j] for j in range(n)]
            rhs = [rp[i] - sum(M[i][j] * t[j] for j in range(n))
                   + sum(M[i][j] * d[j] * rd[j] for j in range(n))
                   for i in range(m)]
            dy = _chol_solve(L, rhs)
            ds = [rd[j] - sum(M[i][j] * dy[i] for i in range(m))
                  for j in range(n)]
            dx = [t[j] - d[j] * ds[j] for j in range(n)]
            return dx, dy, ds

        def alpha(v, dv):
            a = 1.0
            for j in range(len(v)):
                if dv[j] < 0:
                    a = min(a, -v[j] / dv[j])
            return a

        dxa, dya, dsa = step([-x[j] * s[j] for j in range(n)])
        ap = min(1.0, alpha(x, dxa))
        ad = min(1.0, alpha(s, dsa))
        mu_aff = sum((x[j] + ap * dxa[j]) * (s[j] + ad * dsa[j])
                     for j in range(n)) / n
        sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0
        r3 = [-x[j] * s[j] - dxa[j] * dsa[j] + sigma * mu
              for j in range(n)]
        dx, dy, ds = step(r3)
        ap = min(1.0, 0.99 * alpha(x, dx))
        ad = min(1.0, 0.99 * alpha(s, ds))
        x = [x[j] + ap * dx[j] for j in range(n)]
        s = [s[j] + ad * ds[j] for j in range(n)]
        y = [y[i] + ad * dy[i] for i in range(m)]
    return {"x": x, "y": y, "s": s, "iterations": int(max_iter),
            "gap": sum(x[j] * s[j] for j in range(n)),
            "converged": False, "mu_history": hist,
            "primal_residual": float("nan"),
            "dual_residual": float("nan")}


def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
             upper=None, method="auto", maximise=False, tol=1e-10,
             max_iter=200, rule="bland"):
    r"""Solve a linear program by whichever method is asked for.

    ``auto`` uses simplex, which is exact on integral data and gives a
    vertex; interior point is the choice when the analytic centre of
    the optimal face is wanted instead.
    """
    if method not in METHODS:
        raise ValueError("linprm: method must be one of %s, got %r"
                         % (", ".join(METHODS), method))
    if method in ("simplex", "auto"):
        r = _simplex_solve(c, A_ub, b_ub, A_eq, b_eq, upper, rule,
                           maximise, 10000)
        out = dict(r)
        out["method"] = ("two-phase primal simplex (Dantzig 1963)"
                         if r["status"] == "optimal" else out.get(
                             "method", "simplex"))
        out["solver"] = "simplex"
        return RichResult(payload=out)
    sign = -1.0 if maximise else 1.0
    sf = standard_form([sign * float(v) for v in c], A_ub, b_ub,
                       A_eq, b_eq, upper)
    r = interior_point(sf["c"], sf["A"], sf["b"], tol, max_iter)
    n = sf["n_original"]
    x = [max(0.0, v) for v in r["x"][:n]]
    fun = sign * sum(sf["c"][j] * r["x"][j]
                     for j in range(len(sf["c"])))
    return RichResult(payload={
        "estimate": x, "x": x, "fun": fun,
        "duals": [sign * v for v in r["y"]],
        "slack": r["x"][n:],
        "status": "optimal" if r["converged"] else "no_convergence",
        "gap": r["gap"], "iterations": r["iterations"],
        "primal_residual": r["primal_residual"],
        "dual_residual": r["dual_residual"],
        "maximise": bool(maximise), "solver": "interior_point",
        "n_original": n, "n_slack": sf["n_slack"],
        "method": "Mehrotra (1992) predictor-corrector primal-dual "
                  "interior point",
    })


def linear_program(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                   upper=None, method="auto", **kw):
    r"""Entry point: see :func:`solve_lp`."""
    return solve_lp(c, A_ub, b_ub, A_eq, b_eq, upper, method, **kw)
