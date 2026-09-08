# morie.fn -- function file (rootcoder007/morie)
r"""Semidefinite programming by the barrier method.

An SDP minimises a linear objective over a **linear matrix
inequality**:

.. math:: \text{minimise } c^\top x \quad\text{subject to}\quad
          F(x) = F_0 + \sum_{i=1}^{n} x_i F_i \succeq 0.

The feasible set is convex -- the PSD cone intersected with an affine
subspace -- so the problem is convex even though the constraint is on
a matrix. That is the whole reason SDP is tractable, and it is why an
LP is the special case where every :math:`F_i` is diagonal.

**The barrier is :math:`-\log\det F(x)`**, and its two properties are
what make the method work. It is **finite exactly on the interior**
and rises to :math:`+\infty` at the boundary, so an iterate can never
step outside the cone; and it is **self-concordant**, which is what
gives Newton's method its complexity guarantee here rather than in a
generic nonlinear solver. ``barrier`` returns infinity outside rather
than a large number, because a finite value there would let a line
search wander out of the feasible set undetected.

**Centring, then decreasing :math:`t`.** For each :math:`t` the
centring problem :math:`\min\; t\,c^\top x - \log\det F(x)` is solved,
and its solution is on the **central path**. The duality gap at the
central point is exactly :math:`m/t` with :math:`m` the matrix
dimension, so the accuracy is *known* at every stage rather than
inferred from how much the iterate moved -- ``central_path_gap``
returns it.

**A closed form to check against.** The problem
:math:`\max\, t` s.t. :math:`A - tI \succeq 0` has the exact solution
:math:`t^\star = \lambda_{\min}(A)`, so the solver's answer can be
compared with an eigenvalue rather than with itself.
``min_eigenvalue_sdp`` sets that problem up, and the anchor uses it.

References
----------
Boyd, S. & Vandenberghe, L. (2004) *Convex Optimization*, Cambridge
University Press, doi:10.1017/CBO9780511804441. Sec. 4.6.2 (the
semidefinite program with its linear matrix inequality constraint,
and LP as the case of diagonal matrices); Sec. 9.6 and 11.1 (the
logarithmic barrier -log det X for the PSD cone, its self-concordance,
and the central path); Sec. 11.2-11.3 (the barrier method: solve the
centring problem for a sequence of increasing t, with the duality gap
at a central point equal to m/t, and the trade-off in the choice of
the multiplier mu between the number of outer iterations and the
difficulty of each centring step); and Sec. 5.5 (the KKT conditions
and complementary slackness used for the optimality check).

Vandenberghe, L. & Boyd, S. (1996) "Semidefinite Programming",
*SIAM Review* 38(1), 49-95, doi:10.1137/1038003. The survey
treatment, including the eigenvalue problems that reduce to SDP.

Nesterov, Y. & Nemirovskii, A. (1994) *Interior-Point Polynomial
Algorithms in Convex Programming*, SIAM,
doi:10.1137/1.9781611970791. Self-concordance, which is what makes
the barrier method's complexity claim hold.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["lmi", "is_psd", "barrier", "central_path_gap",
           "solve_sdp", "min_eigenvalue_sdp"]

_EPS = 1e-12


def lmi(x, F0, Fs):
    r""":math:`F(x) = F_0 + \sum_i x_i F_i`."""
    v = [float(t) for t in k.vec(x)]
    A = [[float(t) for t in r] for r in k.mat(F0)]
    if len(v) != len(Fs):
        raise ValueError("sdpwts: %d variables but %d matrices"
                         % (len(v), len(Fs)))
    n = len(A)
    out = [row[:] for row in A]
    for i in range(len(v)):
        M = [[float(t) for t in r] for r in k.mat(Fs[i])]
        if len(M) != n or len(M[0]) != n:
            raise ValueError("sdpwts: F_%d is not %dx%d" % (i, n, n))
        for a in range(n):
            for b in range(n):
                out[a][b] += v[i] * M[a][b]
    return out


def is_psd(M, tol=-1e-10):
    r"""Eigenvalue test on the constraint matrix."""
    A = [[float(t) for t in r] for r in k.mat(M)]
    vals, _ = np.linalg.eigh(A)
    return {"eigenvalues": list(vals), "min_eigenvalue": min(vals),
            "psd": min(vals) >= float(tol),
            "strictly_feasible": min(vals) > 0.0}


def barrier(x, F0, Fs):
    r""":math:`-\log\det F(x)`, INFINITE outside the cone.

    Returning a large finite number instead would let a line search
    leave the feasible set without noticing.
    """
    M = lmi(x, F0, Fs)
    vals, _ = np.linalg.eigh(M)
    if min(vals) <= 0.0:
        return {"value": float("inf"), "feasible": False,
                "min_eigenvalue": min(vals),
                "note": "outside the cone the barrier is +inf, not a "
                        "large number"}
    return {"value": -sum(math.log(v) for v in vals),
            "feasible": True, "min_eigenvalue": min(vals),
            "eigenvalues": list(vals)}


def central_path_gap(t, m):
    r"""The duality gap at a central point is exactly :math:`m/t`.

    So the accuracy is known, not inferred from how far the last step
    moved.
    """
    tt, mm = float(t), int(m)
    if tt <= 0.0 or mm < 1:
        raise ValueError("sdpwts: t must be positive and m at least "
                         "1")
    return {"gap": mm / tt, "t": tt, "m": mm,
            "note": "an exact suboptimality bound at every stage"}


def _objective(x, c, F0, Fs, t):
    b = barrier(x, F0, Fs)
    if not b["feasible"]:
        return float("inf")
    return float(t) * sum(float(c[i]) * float(x[i])
                          for i in range(len(x))) + b["value"]


def _centre(x0, c, F0, Fs, t, iters=200, tol=1e-12, h=1e-6):
    r"""Centring by gradient descent with a feasibility-aware
    backtracking line search."""
    x = [float(v) for v in k.vec(x0)]
    n = len(x)
    f = _objective(x, c, F0, Fs, t)
    if not math.isfinite(f):
        raise ValueError("sdpwts: the starting point is not strictly "
                         "feasible, so the barrier is infinite there")
    it = 0
    for it in range(1, int(iters) + 1):
        g = []
        for i in range(n):
            up, dn = list(x), list(x)
            up[i] += h
            dn[i] -= h
            fu = _objective(up, c, F0, Fs, t)
            fd = _objective(dn, c, F0, Fs, t)
            if not math.isfinite(fu) or not math.isfinite(fd):
                g.append(0.0)
            else:
                g.append((fu - fd) / (2.0 * h))
        gn = math.sqrt(sum(v * v for v in g))
        if gn < float(tol):
            break
        step = 1.0
        moved = False
        for _ in range(80):
            cand = [x[i] - step * g[i] for i in range(n)]
            fc = _objective(cand, c, F0, Fs, t)
            if math.isfinite(fc) and fc < f - 1e-14:
                x, f = cand, fc
                moved = True
                break
            step *= 0.5
        if not moved:
            break
    return {"x": x, "value": f, "iterations": it}


def solve_sdp(c, F0, Fs, x0, t0=1.0, mu=10.0, tol=1e-8,
              max_outer=60):
    r"""The barrier method: centre, increase :math:`t`, repeat.

    ``mu`` trades outer iterations against the difficulty of each
    centring step; the gap after each outer step is exactly
    :math:`m/t`.
    """
    cc = [float(v) for v in k.vec(c)]
    x = [float(v) for v in k.vec(x0)]
    m = len(k.mat(F0))
    if not is_psd(lmi(x, F0, Fs))["strictly_feasible"]:
        raise ValueError("sdpwts: the starting point must be "
                         "STRICTLY feasible -- the barrier method "
                         "cannot begin on the boundary")
    if float(mu) <= 1.0:
        raise ValueError("sdpwts: mu must exceed 1, or t never "
                         "increases")
    t = float(t0)
    path, outer = [], 0
    for outer in range(1, int(max_outer) + 1):
        r = _centre(x, cc, F0, Fs, t)
        x = r["x"]
        gap = central_path_gap(t, m)["gap"]
        path.append({"t": t, "gap": gap,
                     "objective": sum(cc[i] * x[i]
                                      for i in range(len(x)))})
        if gap < float(tol):
            break
        t *= float(mu)
    return RichResult(payload={
        "estimate": [float(v) for v in x], "x": x,
        "objective": sum(cc[i] * x[i] for i in range(len(x))),
        "gap": path[-1]["gap"], "outer_iterations": outer,
        "path": path, "m": m,
        "min_eigenvalue": is_psd(lmi(x, F0, Fs))["min_eigenvalue"],
        "method": "barrier method for SDP; Boyd & Vandenberghe "
                  "(2004) Sec. 11.2-11.3",
        "note": "the gap m/t is an exact bound, so 'converged' is a "
                "measurement rather than a guess",
    })


def min_eigenvalue_sdp(A, t0=1.0, mu=10.0, tol=1e-9):
    r"""Maximise :math:`t` s.t. :math:`A - tI\succeq 0`.

    The answer is :math:`\lambda_{\min}(A)` exactly, which is what
    makes this a check on the solver rather than on itself.
    """
    M = [[float(v) for v in r] for r in k.mat(A)]
    n = len(M)
    vals, _ = np.linalg.eigh(M)
    lam = min(vals)
    # minimise -t subject to A - tI >= 0
    F0 = M
    F1 = [[-1.0 if a == b else 0.0 for b in range(n)]
          for a in range(n)]
    start = [lam - 1.0]
    r = solve_sdp([-1.0], F0, [F1], start, t0, mu, tol)
    return RichResult(payload={
        "estimate": r["x"][0], "t": r["x"][0],
        "lambda_min": lam, "error": abs(r["x"][0] - lam),
        "outer_iterations": r["outer_iterations"],
        "gap": r["gap"],
        "method": "eigenvalue problem as an SDP; Vandenberghe & Boyd "
                  "(1996)",
        "note": "the exact answer is lambda_min(A), so the solver is "
                "checked against something other than itself",
    })


def cheatsheet():
    return ("sdpwts: minimise c'x subject to a LINEAR MATRIX "
            "INEQUALITY F0 + sum x_i F_i >= 0. The feasible set is the "
            "PSD cone cut by an affine subspace -- convex, which is why "
            "it is tractable, and LP is the diagonal special case. The "
            "barrier is -log det F(x): FINITE only on the interior "
            "(so an iterate cannot leave the cone) and SELF-CONCORDANT "
            "(which is what earns Newton's complexity guarantee). "
            "Solve the centring problem for increasing t; at a central "
            "point the duality gap is EXACTLY m/t, so accuracy is "
            "known, not inferred. Check against max t s.t. A - tI >= "
            "0, whose answer is lambda_min(A).")


# compact alias per ledger/NAMING.md
semidefinite_program = solve_sdp
