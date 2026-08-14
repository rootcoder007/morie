# morie.fn -- function file (rootcoder007/morie)
r"""The SVM dual, solved two variables at a time.

The soft-margin SVM's primal has one slack per example; its **dual**

.. math:: \max_\alpha\; \sum_i \alpha_i
          - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_iy_j K(x_i,x_j),
          \qquad 0\le\alpha_i\le C,\quad \sum_i y_i\alpha_i = 0,

is where the kernel enters -- the data appear only through inner
products -- and where the structure is exploitable.

**Why decomposition, and why exactly two.** The Hessian is dense and
:math:`\ell\times\ell`; at :math:`\ell = 10^5` it does not fit in
memory, so a solver must work on a subset at a time. The equality
constraint :math:`\sum y_i\alpha_i = 0` means a single variable cannot
move alone without breaking it -- **two** is the smallest working set
that can move at all, and for two the subproblem has a closed-form
solution, so no inner QP solver is needed. ``solve_pair`` is that
closed form.

**The pair is chosen by maximal violation, not at random.** With
:math:`\nabla_i` the gradient, the KKT conditions say the optimum is
reached when

.. math:: \max_{i\in I_{up}} -y_i\nabla_i \;\le\;
          \min_{j\in I_{low}} -y_j\nabla_j,

so the violating pair with the largest gap is the natural choice, and
that same quantity is the **stopping criterion** -- the optimality gap
is measured, not guessed at from an iteration count.
``kkt_violation`` returns it.

**Clipping is where the box constraint lives.** The analytic step is
truncated to :math:`[L, H]`, bounds that depend on whether the two
labels agree. Getting that branch wrong yields a solver that still
converges -- to the wrong answer -- which is why the anchor checks the
constraint set directly rather than only the objective.

**The bias comes from the free support vectors.** For
:math:`0<\alpha_i<C` the margin is tight, so each such point gives
:math:`b` exactly; averaging them is a numerical convenience, not a
definition, and when none are free the value is bracketed instead.

References
----------
Chang, C.-C. & Lin, C.-J. (2011) "LIBSVM: A Library for Support
Vector Machines", *ACM Transactions on Intelligent Systems and
Technology* 2(3), Article 27, doi:10.1145/1961189.1961199. The dual
formulation with the box and equality constraints; the decomposition
method working on two variables at a time because the density of the
Hessian makes the full problem infeasible in memory; working-set
selection by the maximal violating pair from the sets I_up and I_low;
the stopping condition expressed as the gap between the two extreme
gradient terms; and the recovery of b from the free support vectors.

Cortes, C. & Vapnik, V. (1995) "Support-Vector Networks", *Machine
Learning* 20(3), 273-297, doi:10.1007/BF00994018. [PDF supplied by
Vee.] The soft-margin formulation, its dual, and the support vectors
as the examples with non-zero multipliers.

Platt, J. C. (1998) "Sequential Minimal Optimization: A Fast
Algorithm for Training Support Vector Machines", Microsoft Research
Technical Report MSR-TR-98-14. [PDF supplied by Vee after every
fetch route returned 403.] The two-variable decomposition: that the
smallest possible optimisation problem involves two multipliers,
because the linear equality constraint forces them to move together,
and that this subproblem is solved analytically so no inner QP solver
is invoked at all.

Boyd, S. & Vandenberghe, L. (2004) *Convex Optimization*, Cambridge
University Press, doi:10.1017/CBO9780511804441. Sec. 5.2 and 5.5 for
the Lagrange dual and the KKT conditions the stopping rule rests on.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["kernel_matrix", "dual_objective", "solve_pair",
           "kkt_violation", "smo", "recover_bias"]

_EPS = 1e-12
_TAU = 1e-12


def kernel_matrix(X, kernel="linear", gamma=1.0, degree=3, coef0=0.0):
    r"""The Gram matrix. The data enter ONLY through this."""
    M = [[float(v) for v in r] for r in k.mat(X)]
    n = len(M)

    def kf(a, b):
        d = sum(a[t] * b[t] for t in range(len(a)))
        if kernel == "linear":
            return d
        if kernel == "poly":
            return (float(gamma) * d + float(coef0)) ** int(degree)
        if kernel == "rbf":
            s = sum((a[t] - b[t]) ** 2 for t in range(len(a)))
            return math.exp(-float(gamma) * s)
        raise ValueError("svmopt: kernel must be linear, poly or "
                         "rbf, got %r" % (kernel,))

    return [[kf(M[i], M[j]) for j in range(n)] for i in range(n)]


def dual_objective(alpha, y, K):
    r""":math:`\sum_i\alpha_i - \frac12\sum_{ij}\alpha_i\alpha_j
    y_iy_jK_{ij}`."""
    a = [float(v) for v in k.vec(alpha)]
    yy = [float(v) for v in k.vec(y)]
    n = len(a)
    q = 0.0
    for i in range(n):
        if a[i] == 0.0:
            continue
        for j in range(n):
            q += a[i] * a[j] * yy[i] * yy[j] * K[i][j]
    return sum(a) - 0.5 * q


def _bounds(i, j, a, y, C):
    if y[i] != y[j]:
        L = max(0.0, a[j] - a[i])
        Hh = min(C, C + a[j] - a[i])
    else:
        L = max(0.0, a[i] + a[j] - C)
        Hh = min(C, a[i] + a[j])
    return L, Hh


def solve_pair(i, j, alpha, y, K, grad, C):
    r"""The two-variable subproblem, in closed form.

    Two is the smallest working set the equality constraint allows to
    move, and at two the QP has an analytic solution -- which is the
    whole reason the decomposition is cheap.
    """
    a = [float(v) for v in k.vec(alpha)]
    if i == j:
        raise ValueError("svmopt: the working set must contain two "
                         "DIFFERENT indices")
    L, Hh = _bounds(i, j, a, y, C)
    if Hh <= L + _EPS:
        return {"alpha": a, "moved": 0.0, "clipped": True,
                "L": L, "H": Hh,
                "note": "the box leaves no room for this pair"}
    # eta and the step both carry the LABELS: the working set moves
    # along the direction (y_i, -y_j) that keeps sum(y a) fixed, so
    # y_i y_j appears in the curvature and the numerator is the KKT
    # gap itself. Dropping the labels gives a zero step whenever the
    # two gradients happen to agree -- which they do at the start,
    # where every gradient is -1.
    eta = K[i][i] + K[j][j] - 2.0 * y[i] * y[j] * K[i][j]
    if eta <= _TAU:
        eta = _TAU
    step = ((-y[i] * grad[i]) - (-y[j] * grad[j])) / eta
    aj_new = a[j] - y[j] * step
    aj_cl = min(max(aj_new, L), Hh)
    delta = aj_cl - a[j]
    out = list(a)
    out[j] = aj_cl
    out[i] = a[i] - y[i] * y[j] * delta
    return {"alpha": out, "moved": abs(delta),
            "clipped": abs(aj_cl - aj_new) > _EPS, "L": L, "H": Hh,
            "eta": eta, "step": step}


def kkt_violation(alpha, y, grad, C):
    r"""The optimality gap: :math:`\max_{I_{up}} -y\nabla -
    \min_{I_{low}} -y\nabla`.

    This is the stopping criterion AND the working-set rule, so
    convergence is measured rather than assumed after N iterations.
    """
    a = [float(v) for v in k.vec(alpha)]
    yy = [float(v) for v in k.vec(y)]
    up, low = [], []
    for t in range(len(a)):
        if (yy[t] > 0 and a[t] < C - _EPS) or \
                (yy[t] < 0 and a[t] > _EPS):
            up.append(t)
        if (yy[t] > 0 and a[t] > _EPS) or \
                (yy[t] < 0 and a[t] < C - _EPS):
            low.append(t)
    if not up or not low:
        return {"gap": 0.0, "i": None, "j": None,
                "note": "no violating pair exists"}
    i = max(up, key=lambda t: -yy[t] * grad[t])
    j = min(low, key=lambda t: -yy[t] * grad[t])
    return {"gap": (-yy[i] * grad[i]) - (-yy[j] * grad[j]),
            "i": i, "j": j, "n_up": len(up), "n_low": len(low)}


def recover_bias(alpha, y, grad, C):
    r""":math:`b` from the FREE support vectors, where the margin is
    tight."""
    a = [float(v) for v in k.vec(alpha)]
    yy = [float(v) for v in k.vec(y)]
    free = [t for t in range(len(a)) if _EPS < a[t] < C - _EPS]
    if free:
        vals = [-yy[t] * grad[t] for t in free]
        return {"b": sum(vals) / len(vals), "n_free": len(free),
                "bracketed": False,
                "spread": max(vals) - min(vals)}
    v = kkt_violation(a, yy, grad, C)
    lo = -yy[v["j"]] * grad[v["j"]] if v["j"] is not None else 0.0
    hi = -yy[v["i"]] * grad[v["i"]] if v["i"] is not None else 0.0
    return {"b": 0.5 * (lo + hi), "n_free": 0, "bracketed": True,
            "note": "no free support vector, so b is only bracketed"}


def smo(y, K, C=1.0, tol=1e-8, max_iter=20000):
    r"""Decomposition on the maximal violating pair until the KKT gap
    closes."""
    yy = [float(v) for v in k.vec(y)]
    n = len(yy)
    if any(v not in (-1.0, 1.0) for v in yy):
        raise ValueError("svmopt: labels must be -1 or +1")
    if len(K) != n or len(K[0]) != n:
        raise ValueError("svmopt: the kernel matrix is %dx%d for %d "
                         "labels" % (len(K), len(K[0]), n))
    if float(C) <= 0.0:
        raise ValueError("svmopt: C must be positive")
    a = [0.0] * n
    grad = [-1.0] * n
    it, gap = 0, float("inf")
    for it in range(1, int(max_iter) + 1):
        v = kkt_violation(a, yy, grad, C)
        gap = v["gap"]
        if v["i"] is None or gap <= float(tol):
            break
        r = solve_pair(v["i"], v["j"], a, yy, K, grad, C)
        if r["moved"] <= _EPS:
            break
        di = r["alpha"][v["i"]] - a[v["i"]]
        dj = r["alpha"][v["j"]] - a[v["j"]]
        a = r["alpha"]
        for t in range(n):
            grad[t] += (yy[t] * yy[v["i"]] * K[t][v["i"]] * di
                        + yy[t] * yy[v["j"]] * K[t][v["j"]] * dj)
    b = recover_bias(a, yy, grad, C)
    sv = [t for t in range(n) if a[t] > _EPS]
    return RichResult(payload={
        "estimate": a, "alpha": a, "b": b["b"], "gap": gap,
        "iterations": it, "converged": gap <= float(tol),
        "support_vectors": sv, "n_sv": len(sv),
        "n_free": b["n_free"],
        "equality_residual": sum(a[t] * yy[t] for t in range(n)),
        "objective": dual_objective(a, yy, K),
        "method": "two-variable decomposition on the maximal "
                  "violating pair; Chang & Lin (2011)",
        "note": "the KKT gap is both the working-set rule and the "
                "stopping criterion",
    })


def cheatsheet():
    return ("svmopt: the SVM DUAL is where the kernel enters and where "
            "the structure is exploitable -- max sum(a) - 0.5 a'Qa "
            "subject to 0 <= a <= C and sum(y a) = 0. The Hessian is "
            "dense and l x l, so decompose; the EQUALITY constraint "
            "means one variable cannot move alone, so TWO is the "
            "smallest workable set -- and at two the subproblem is "
            "closed form. Choose the pair by MAXIMAL KKT VIOLATION, "
            "which is also the stopping criterion, so convergence is "
            "measured not assumed. Clip to [L,H], whose branch depends "
            "on whether the labels agree -- get it wrong and the "
            "solver still converges, to the wrong answer. b comes from "
            "the FREE support vectors.")


# compact alias per ledger/NAMING.md
svm_dual_qp = smo
