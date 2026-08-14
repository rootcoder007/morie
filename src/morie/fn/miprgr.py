# morie.fn -- function file (rootcoder007/morie)
r"""Branch and bound: solve the relaxation, then split on a fraction.

An integer program's relaxation is easy and its answer is usually
**fractional**. Rounding it is not a fix: the rounded point can be
infeasible, and when it is feasible it can be strictly worse than the
integer optimum. ``round_relaxation`` produces that failure rather
than describing it.

**The two halves of the method.**

*Bound.* The relaxation's value is a valid bound on every integer
solution below that node -- for a maximisation, no descendant can beat
it. So once an integer solution is in hand, any node whose relaxation
is no better can be **discarded without being explored**. That is
what makes the search finite in practice rather than in principle.

*Branch.* Land and Doig's original scheme enumerated the values of an
integer variable. **Dakin's** modification is the one used here and
almost everywhere since: pick a variable whose relaxed value
:math:`v` is fractional and split on the *dichotomy*

.. math:: x_j \le \lfloor v \rfloor \quad\text{or}\quad
          x_j \ge \lceil v \rceil,

which excludes the fractional point, covers every integer point
between the two branches, and -- unlike enumeration -- keeps each node
a linear program of the same kind, so a binary tree of LPs is all that
is ever solved.

**Pruning must not change the answer.** ``branch_and_bound`` reports
the node count, and the anchor runs it with pruning disabled to check
the optimum is identical and the node count strictly larger -- the
property that makes a bound *safe* rather than merely fast.

**The incumbent is the only thing that makes bounding work**, so a
good first solution matters; depth-first descent finds one early,
which is why it is the default order here.

References
----------
Land, A. H. & Doig, A. G. (1960) "An Automatic Method of Solving
Discrete Programming Problems", *Econometrica* 28(3), 497-520,
doi:10.2307/1910129. [PDF supplied by Vee.] The original
branch-and-bound scheme for discrete programming: solving a sequence
of continuous relaxations, using their values as bounds on the
attainable integer objective, and systematically subdividing the
feasible region so that the search terminates.

Dakin, R. J. (1965) "A tree-search algorithm for mixed integer
programming problems", *The Computer Journal* 8(3), 250-255,
doi:10.1093/comjnl/8.3.250. [PDF supplied by Vee.] The dichotomous
branching used here: rather than enumerating values of an integer
variable, add the two mutually exclusive constraints x_j <= floor(v)
and x_j >= ceil(v) to the relaxation, which excludes the fractional
solution while retaining every integer point, and keeps every node a
linear program so the search is a binary tree.

Mehrotra, S. (1992) "On the Implementation of a Primal-Dual Interior
Point Method", *SIAM Journal on Optimization* 2(4), 575-601,
doi:10.1137/0802028. The LP solver used at each node; implemented in
:mod:`mehtad`.

Dantzig, G. B. (1963) *Linear Programming and Extensions*, Princeton
University Press, doi:10.1515/9781400884179. The relaxation being
solved.
"""

import math

from . import _array_core as np
from . import _s03core as k
from . import mehtad as ip
from ._richresult import RichResult

__all__ = ["solve_relaxation", "fractional_variable",
           "round_relaxation", "branch_and_bound",
           "enumerate_integer"]

_EPS = 1e-7


def _simplex(A, b, c, tol=1e-9, max_iter=20000):
    r"""Two-phase simplex with Bland's rule, maximising c'x.

    An interior-point method cannot serve here: a branch such as
    x_j >= 4 can collapse the region to a SINGLE POINT, which has no
    strict interior, and the solver then reports "no optimum" for a
    node that is perfectly feasible -- pruning a subtree that contains
    the answer. The simplex works on vertices, so a point polytope is
    an ordinary case. Bland's rule (lowest index in, lowest basic
    index out) is chosen over the steepest-descent rule because it is
    the one with a termination proof; cycling on a degenerate node
    would hang the whole tree search.
    """
    m = len(A)
    n = len(A[0]) if m else 0
    rows = [[float(v) for v in A[i]] for i in range(m)]
    rhs = [float(v) for v in b]
    for i in range(m):
        if rhs[i] < 0.0:
            rows[i] = [-v for v in rows[i]]
            rhs[i] = -rhs[i]
            rows[i] = rows[i] + [0.0] * m
            rows[i][n + i] = -1.0
        else:
            rows[i] = rows[i] + [0.0] * m
            rows[i][n + i] = 1.0
    need_art = [i for i in range(m) if rows[i][n + i] < 0.0]
    na = len(need_art)
    width = n + m + na
    T = []
    for i in range(m):
        T.append(rows[i] + [0.0] * na + [rhs[i]])
    basis = [n + i for i in range(m)]
    for a, i in enumerate(need_art):
        T[i][n + m + a] = 1.0
        basis[i] = n + m + a

    def reduced(obj):
        z = list(obj)
        for i in range(m):
            f = z[basis[i]]
            if f != 0.0:
                for j in range(width + 1):
                    z[j] -= f * T[i][j]
        return z

    def pivot(pr, pc):
        pv = T[pr][pc]
        T[pr] = [v / pv for v in T[pr]]
        for i in range(m):
            if i != pr and T[i][pc] != 0.0:
                f = T[i][pc]
                for j in range(width + 1):
                    T[i][j] -= f * T[pr][j]
        basis[pr] = pc

    def run(obj, allowed):
        for _ in range(int(max_iter)):
            z = reduced(obj)
            enter = -1
            for j in allowed:
                if z[j] < -tol:      # Bland: lowest eligible index
                    enter = j
                    break
            if enter < 0:
                return True
            ratio, leave = None, -1
            for i in range(m):
                if T[i][enter] > tol:
                    r = T[i][-1] / T[i][enter]
                    if (ratio is None or r < ratio - tol
                            or (abs(r - ratio) <= tol and leave >= 0
                                and basis[i] < basis[leave])):
                        ratio, leave = r, i
            if leave < 0:
                return False          # unbounded
            pivot(leave, enter)
        return False

    if na:
        phase1 = [0.0] * (width + 1)
        for a in range(na):
            phase1[n + m + a] = 1.0
        if not run(phase1, list(range(n + m))):
            return {"feasible": False, "x": None, "value": None}
        infeas = sum(T[i][-1] for i in range(m)
                     if basis[i] >= n + m)
        if infeas > 1e-7:
            return {"feasible": False, "x": None, "value": None}
        for i in range(m):            # drive artificials out
            if basis[i] >= n + m:
                for j in range(n + m):
                    if abs(T[i][j]) > tol:
                        pivot(i, j)
                        break
    phase2 = [0.0] * (width + 1)
    for j in range(n):
        phase2[j] = -float(c[j])
    if not run(phase2, list(range(n + m))):
        return {"feasible": False, "x": None, "value": None}
    x = [0.0] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i][-1]
    return {"feasible": True, "x": x,
            "value": sum(float(c[j]) * x[j] for j in range(n))}


def _standard_form(A, b, c, bounds, n):
    r"""Add a slack per inequality, including the branch bounds."""
    rows, rhs = [], []
    for i in range(len(A)):
        rows.append([float(v) for v in A[i]])
        rhs.append(float(b[i]))
    for (j, sense, val) in bounds:
        r = [0.0] * n
        r[int(j)] = 1.0 if sense == "le" else -1.0
        rows.append(r)
        rhs.append(float(val) if sense == "le" else -float(val))
    m = len(rows)
    full = []
    for i in range(m):
        full.append(rows[i] + [1.0 if t == i else 0.0
                               for t in range(m)])
    return full, rhs, [float(v) for v in c] + [0.0] * m


def solve_relaxation(A, b, c, bounds=(), n=None, maximise=True,
                     solver="simplex"):
    r"""The LP relaxation at one node.

    ``solver="simplex"`` is the default and the only one safe for
    branch and bound; ``"interior"`` routes to :mod:`mehtad` and is
    kept because it is the better method on large sparse LPs -- but a
    node whose region is a single point has no interior, and treating
    that as infeasible would prune the optimum.
    """
    nn = int(n) if n is not None else len(c)
    if solver == "simplex":
        rows, rhs2 = [], []
        for i in range(len(A)):
            rows.append([float(v) for v in A[i]])
            rhs2.append(float(b[i]))
        for (j, sense, val) in bounds:
            r = [0.0] * nn
            r[int(j)] = 1.0 if sense == "le" else -1.0
            rows.append(r)
            rhs2.append(float(val) if sense == "le"
                        else -float(val))
        sgn = 1.0 if maximise else -1.0
        out = _simplex(rows, rhs2, [sgn * float(v) for v in c])
        if not out["feasible"]:
            return {"feasible": False, "x": None, "value": None,
                    "note": "the relaxation is infeasible, so every "
                            "integer point below this node is too"}
        x = [max(0.0, v) for v in out["x"]]
        return {"feasible": True, "x": x,
                "value": sum(float(c[j]) * x[j] for j in range(nn)),
                "note": "a valid BOUND on every integer point below "
                        "this node"}
    if solver != "interior":
        raise ValueError("miprgr: solver must be simplex or "
                         "interior, got %r" % (solver,))
    M, rhs, cc = _standard_form(A, b, c, list(bounds), nn)
    obj = [-v for v in cc] if maximise else list(cc)
    try:
        r = ip.solve_lp(M, rhs, obj, tol=1e-10, max_iter=200)
    except ValueError:
        return {"feasible": False, "x": None, "value": None,
                "note": "the relaxation is infeasible, so every "
                        "integer point below this node is too"}
    if not r["converged"]:
        return {"feasible": False, "x": None, "value": None,
                "note": "no interior optimum found"}
    x = [max(0.0, r["x"][j]) for j in range(nn)]
    # Feasibility is checked on the SOLUTION, not on the sign of the
    # right-hand side: a >= branch is written -x_j + slack = -val, so
    # its rhs is negative by construction. Rejecting negative rhs
    # would silently prune every >= branch -- and the search would
    # still terminate and still report an optimum, just the wrong
    # one.
    for i in range(len(A)):
        if sum(float(A[i][j]) * x[j]
               for j in range(nn)) > float(b[i]) + 1e-6:
            return {"feasible": False, "x": None, "value": None,
                    "note": "the relaxation violates an original "
                            "constraint"}
    for (j, sense, val) in bounds:
        if sense == "le" and x[int(j)] > float(val) + 1e-6:
            return {"feasible": False, "x": None, "value": None,
                    "note": "branch bound violated"}
        if sense == "ge" and x[int(j)] < float(val) - 1e-6:
            return {"feasible": False, "x": None, "value": None,
                    "note": "branch bound violated"}
    val = sum(float(c[j]) * x[j] for j in range(nn))
    return {"feasible": True, "x": x, "value": val,
            "note": "a valid BOUND on every integer point below this "
                    "node"}


def fractional_variable(x, integer_vars, tol=_EPS):
    r"""The most fractional integer variable, or None if all are
    integral."""
    best, gap = None, 0.0
    for j in integer_vars:
        v = float(x[j])
        f = abs(v - round(v))
        if f > tol and f > gap:
            best, gap = j, f
    return {"index": best, "fractionality": gap,
            "integral": best is None}


def round_relaxation(x, A, b, integer_vars):
    r"""Round the relaxation and check what it gives.

    Usually infeasible, and when feasible often strictly worse -- the
    reason branching exists.
    """
    xr = [float(v) for v in x]
    for j in integer_vars:
        xr[j] = float(round(xr[j]))
    viol = []
    for i in range(len(A)):
        lhs = sum(float(A[i][j]) * xr[j] for j in range(len(xr)))
        if lhs > float(b[i]) + _EPS:
            viol.append({"row": i, "lhs": lhs, "rhs": float(b[i])})
    return {"x": xr, "feasible": not viol, "violations": viol,
            "note": "rounding is not a substitute for branching"}


def enumerate_integer(A, b, c, integer_vars, upper=10,
                      maximise=True):
    r"""Brute force over a small integer box, for checking.

    Not a method -- a way to know the right answer independently of
    the search.
    """
    n = len(c)
    best, best_x = (-float("inf") if maximise else float("inf")), None
    stack = [[]]
    while stack:
        pre = stack.pop()
        if len(pre) == n:
            ok = all(sum(float(A[i][j]) * pre[j]
                         for j in range(n)) <= float(b[i]) + _EPS
                     for i in range(len(A)))
            if ok:
                val = sum(float(c[j]) * pre[j] for j in range(n))
                if (maximise and val > best) or \
                        (not maximise and val < best):
                    best, best_x = val, list(pre)
            continue
        j = len(pre)
        rng = range(0, int(upper) + 1) if j in integer_vars else \
            range(0, int(upper) + 1)
        for v in rng:
            stack.append(pre + [float(v)])
    return {"value": best, "x": best_x,
            "note": "exhaustive over the box, so the search can be "
                    "checked against something other than itself"}


def branch_and_bound(A, b, c, integer_vars, maximise=True,
                     prune=True, max_nodes=5000, solver="simplex"):
    r"""Dakin's Fig. 2, step for step, with the marked LIST.

    The list is the part worth copying. Each entry holds one variable
    and one bound constraint plus a **marker** meaning "the
    alternative branch from here has already been taken". Backtracking
    walks up erasing marked entries; at the first unmarked one it
    removes that constraint, adds the alternative, marks the entry and
    descends again. So the store is the current *path*, not the set of
    open nodes -- Dakin's own example has nine solutions and a list
    that never exceeds three entries, and ``max_list_length`` reports
    that.

    The eleven numbered steps of Fig. 2 are marked in the code.
    ``prune=False`` disables the bound test only, so it can be shown
    not to change the answer.
    """
    n = len(c)
    I = sorted(set(int(v) for v in integer_vars))
    if any(j < 0 or j >= n for j in I):
        raise ValueError("miprgr: an integer index is outside the "
                         "variable set")
    better = ((lambda a, bb: a > bb + _EPS) if maximise
              else (lambda a, bb: a < bb - _EPS))
    incumbent = (-float("inf")) if maximise else float("inf")
    inc_x = None
    lst = []                                   # step 1: list empty
    nodes, pruned, max_len = 0, 0, 0
    root_bound = None
    while nodes < int(max_nodes):
        bounds = [(e["var"], e["sense"], e["value"]) for e in lst]
        rel = solve_relaxation(A, b, c, bounds, n, maximise,
                               solver)          # step 2
        nodes += 1
        max_len = max(max_len, len(lst))
        if root_bound is None and rel["feasible"]:
            root_bound = rel["value"]
        descend = False
        if rel["feasible"]:                     # step 3
            cut = (prune and inc_x is not None
                   and not better(rel["value"], incumbent))
            if cut:
                pruned += 1
            else:
                fv = fractional_variable(rel["x"], I)   # step 4
                if fv["integral"]:
                    if better(rel["value"], incumbent):  # step 5
                        incumbent = rel["value"]
                        inc_x = [float(round(v)) if j in I
                                 else float(v)
                                 for j, v in enumerate(rel["x"])]
                else:                            # step 6
                    j = fv["index"]
                    v = rel["x"][j]
                    lst.append({"var": j, "sense": "le",
                                "value": float(math.floor(v)),
                                "alt": ("ge",
                                        float(math.ceil(v))),
                                "marked": False})
                    descend = True
        if descend:
            continue
        while True:                              # step 7
            if not lst:
                return RichResult(payload={      # step 8
                    "estimate": incumbent if inc_x is not None
                    else None,
                    "value": incumbent if inc_x is not None
                    else None,
                    "x": inc_x, "feasible": inc_x is not None,
                    "nodes": nodes, "pruned": pruned,
                    "pruning": bool(prune),
                    "max_list_length": max_len,
                    "root_bound": root_bound,
                    "method": "branch and bound; Land & Doig (1960), "
                              "Dakin (1965) Fig. 2",
                    "note": "the list holds the current PATH, so its "
                            "length is the tree depth, not the "
                            "number of open nodes",
                })
            last = lst[-1]
            if last["marked"]:                   # step 9 -> 10
                lst.pop()
                continue
            sense, val = last["alt"]             # step 11
            last["alt"] = (last["sense"], last["value"])
            last["sense"], last["value"] = sense, val
            last["marked"] = True
            break
    return RichResult(payload={
        "estimate": incumbent if inc_x is not None else None,
        "value": incumbent if inc_x is not None else None,
        "x": inc_x, "feasible": inc_x is not None, "nodes": nodes,
        "pruned": pruned, "pruning": bool(prune),
        "max_list_length": max_len, "root_bound": root_bound,
        "truncated": True,
        "method": "branch and bound; Land & Doig (1960), Dakin "
                  "(1965) Fig. 2",
        "note": "node limit reached, so the result is NOT proven "
                "optimal",
    })


def cheatsheet():
    return ("miprgr: the LP relaxation is easy and usually FRACTIONAL, "
            "and ROUNDING is not a fix -- the rounded point is often "
            "infeasible and, when feasible, often strictly worse. "
            "BOUND: the relaxation's value bounds every integer point "
            "below that node, so once an incumbent exists any node no "
            "better than it can be discarded UNEXPLORED. BRANCH: Land "
            "and Doig enumerated values; DAKIN's dichotomy x_j <= "
            "floor(v) OR x_j >= ceil(v) excludes the fractional point, "
            "keeps every integer point, and leaves each node an LP -- "
            "so the whole search is a binary tree of LPs. Pruning must "
            "not change the optimum: run without it and compare.")


# compact alias per ledger/NAMING.md
mixed_integer_bnb = branch_and_bound
