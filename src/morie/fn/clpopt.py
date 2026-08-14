# morie.fn -- function file (rootcoder007/morie)
r"""The simplex method, with the pivot rule made a visible choice.

**The algorithm.** A linear program in standard form -- minimise
:math:`c'x` subject to :math:`Ax = b`, :math:`x \ge 0` -- has its
optimum at a vertex of the feasible polyhedron. The simplex method
walks from vertex to vertex, each step swapping one variable into the
basis and one out, never increasing the objective. Phase I introduces
an artificial variable per row and minimises their sum, which either
finds a feasible vertex or proves there is none; phase II then
optimises the real objective from there.

**The pivot rule is not a detail.** Which variable enters when several
have negative reduced cost decides whether the method terminates at
all:

``dantzig`` takes the most negative reduced cost -- the steepest local
improvement, and the fastest rule in practice. It can also **cycle
forever** on a degenerate problem, revisiting the same vertex through
a loop of bases with no change in objective. Beale's example does
exactly that, and this implementation detects the repeated basis and
says so instead of grinding to an iteration limit.

``bland`` always takes the lowest-index eligible column, and breaks
ratio ties by lowest basic index. Bland proved this terminates -- no
cycling is possible -- at the cost of being slow. It is the default
here, because a correct answer late beats a wrong answer never.

**Duals come out of the same tableau.** The artificial columns are
kept through phase II (blocked from re-entering), so their reduced
costs at optimality are exactly :math:`-y`. The anchor uses the two
things that must then hold: strong duality, :math:`c'x^* = b'y^*` with
no slack at all, and complementary slackness, every constraint either
tight or with a zero dual.

**Degeneracy and multiple optima are reported, not hidden.** A basic
variable at zero means the vertex is degenerate; a non-basic variable
with zero reduced cost means the optimal face is not a single point,
so the returned :math:`x` is *an* optimum rather than *the* optimum.

References
----------
Dantzig, G. B. (1963) *Linear Programming and Extensions*, Princeton
University Press, doi:10.1515/9781400884179. The simplex method, the
standard form and the two-phase construction reproduced above;
Dantzig's rule of entering the most negative reduced cost.

Bland, R. G. (1977) "New finite pivoting rules for the simplex
method", *Mathematics of Operations Research* 2(2), 103-107,
doi:10.1287/moor.2.2.103. The smallest-subscript rule and the proof
that it terminates, which is why cycling is impossible under it.

Forrest, J. & Lougee-Heimer, R. (2005) "CBC user guide", in *Emerging
Theory, Methods, and Applications*, INFORMS TutORials in Operations
Research, 257-277, doi:10.1287/educ.1053.0020, for the COIN-OR
linear-programming solver whose interface this follows. This is a
native reimplementation, not a binding: morie takes no external
dependencies.
"""

from ._richresult import RichResult

__all__ = ["standard_form", "simplex", "linprog", "PIVOT_RULES"]

PIVOT_RULES = ("bland", "dantzig")
_EPS = 1e-9


def _mat(A, name, ncol=None):
    M = [[float(v) for v in row] for row in A]
    if ncol is not None and any(len(r) != ncol for r in M):
        raise ValueError("clpopt: %s has rows of differing length; "
                         "every row needs %d entries" % (name, ncol))
    return M


def standard_form(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                  upper=None):
    r"""Convert an inequality-form program to :math:`Ax = b,\ x \ge 0`.

    Slack variables are appended for ``<=`` rows and for any finite
    upper bound. The mapping back to the original variables is
    returned so the caller never has to reconstruct it.
    """
    cv = [float(v) for v in c]
    n = len(cv)
    if n == 0:
        raise ValueError("clpopt: the objective has no variables")
    rows, rhs = [], []
    n_slack = 0
    if A_ub is not None:
        M = _mat(A_ub, "A_ub", n)
        bb = [float(v) for v in b_ub]
        if len(M) != len(bb):
            raise ValueError("clpopt: A_ub has %d rows but b_ub has "
                             "%d entries" % (len(M), len(bb)))
        for r, v in zip(M, bb):
            rows.append((list(r), v, "ub"))
            n_slack += 1
    if upper is not None:
        for j, u in enumerate(upper):
            if u is None:
                continue
            if j >= n:
                raise ValueError("clpopt: an upper bound was given "
                                 "for variable %d of %d" % (j, n))
            r = [0.0] * n
            r[j] = 1.0
            rows.append((r, float(u), "ub"))
            n_slack += 1
    if A_eq is not None:
        M = _mat(A_eq, "A_eq", n)
        bb = [float(v) for v in b_eq]
        if len(M) != len(bb):
            raise ValueError("clpopt: A_eq has %d rows but b_eq has "
                             "%d entries" % (len(M), len(bb)))
        for r, v in zip(M, bb):
            rows.append((list(r), v, "eq"))
    if not rows:
        raise ValueError("clpopt: the program has no constraints, so "
                         "it is unbounded unless the objective is "
                         "zero")
    m = len(rows)
    A = []
    b = []
    s = 0
    for r, v, kind in rows:
        row = list(r) + [0.0] * n_slack
        if kind == "ub":
            row[n + s] = 1.0
            s += 1
        A.append(row)
        b.append(v)
    # Every right-hand side must be non-negative for phase I.
    for i in range(m):
        if b[i] < 0:
            A[i] = [-v for v in A[i]]
            b[i] = -b[i]
    return {"A": A, "b": b, "c": cv + [0.0] * n_slack,
            "n_original": n, "n_slack": n_slack,
            "row_kinds": [k for _r, _v, k in rows]}


def _pivot(T, row, col):
    p = T[row][col]
    T[row] = [v / p for v in T[row]]
    for i in range(len(T)):
        if i == row:
            continue
        f = T[i][col]
        if f != 0.0:
            T[i] = [T[i][k] - f * T[row][k] for k in range(len(T[i]))]


def _run(T, basis, cols, rule, blocked, max_iter):
    """Pivot to optimality. Returns a status string."""
    m = len(basis)
    seen = set()
    for _ in range(int(max_iter)):
        cand = [j for j in cols
                if j not in blocked and T[m][j] < -_EPS]
        if not cand:
            return "optimal"
        j = min(cand) if rule == "bland" else min(
            cand, key=lambda k: (T[m][k], k))
        ratios = [(T[i][-1] / T[i][j], basis[i], i)
                  for i in range(m) if T[i][j] > _EPS]
        if not ratios:
            return "unbounded"
        _r, _bi, row = min(ratios)
        _pivot(T, row, j)
        basis[row] = j
        key = tuple(sorted(basis))
        if key in seen:
            return "cycling"
        seen.add(key)
    return "iteration_limit"


def simplex(c, A, b, rule="bland", max_iter=10000,
            initial_basis=None):
    r"""Two-phase primal simplex on :math:`Ax = b,\ x \ge 0`.

    ``b`` must be non-negative; :func:`standard_form` arranges that.

    ``initial_basis`` names ``m`` columns already known to give a
    feasible basis, and skips phase I. Beale's cycling example is
    stated from such a basis, and reproducing the cycle requires
    starting there rather than wherever phase I happens to land.
    """
    if rule not in PIVOT_RULES:
        raise ValueError("clpopt: rule must be one of %s, got %r"
                         % (", ".join(PIVOT_RULES), rule))
    cv = [float(v) for v in c]
    n = len(cv)
    M = _mat(A, "A", n)
    bb = [float(v) for v in b]
    m = len(M)
    if m == 0:
        raise ValueError("clpopt: no constraints")
    if len(bb) != m:
        raise ValueError("clpopt: A has %d rows but b has %d entries"
                         % (m, len(bb)))
    if any(v < -_EPS for v in bb):
        raise ValueError("clpopt: every right-hand side must be "
                         "non-negative in standard form")
    total = n + m
    T = [[M[i][j] for j in range(n)]
         + [1.0 if k == i else 0.0 for k in range(m)]
         + [bb[i]] for i in range(m)]
    basis = [n + i for i in range(m)]
    if initial_basis is not None:
        want = [int(j) for j in initial_basis]
        if len(want) != m:
            raise ValueError("clpopt: initial_basis needs %d columns, "
                             "got %d" % (m, len(want)))
        if any(not 0 <= j < n for j in want) or len(set(want)) != m:
            raise ValueError("clpopt: initial_basis must name %d "
                             "distinct structural columns in [0, %d)"
                             % (m, n))
        for i, j in enumerate(want):
            if abs(T[i][j]) <= _EPS:
                for r in range(i + 1, m):
                    if abs(T[r][j]) > _EPS:
                        T[i], T[r] = T[r], T[i]
                        break
                else:
                    raise ValueError("clpopt: the columns of "
                                     "initial_basis are linearly "
                                     "dependent")
            _pivot(T, i, j)
            basis[i] = j
        if any(T[i][-1] < -_EPS for i in range(m)):
            raise ValueError("clpopt: initial_basis is not feasible "
                             "-- it gives a negative basic value")
        obj2 = [0.0] * (total + 1)
        for j in range(n):
            obj2[j] = cv[j]
        for i in range(m):
            f = cv[basis[i]]
            if f != 0.0:
                for k in range(total + 1):
                    obj2[k] -= f * T[i][k]
        T.append(obj2)
        st = _run(T, basis, range(n), rule, set(range(n, total)),
                  max_iter)
        if st in ("cycling", "iteration_limit"):
            return _fail(st, rule, "phase 2")
        if st == "unbounded":
            return RichResult(payload={
                "estimate": None, "status": "unbounded", "x": None,
                "fun": None, "rule": rule,
                "message": "the objective decreases without bound "
                           "along a feasible ray",
                "method": "primal simplex (Dantzig 1963) from a "
                          "given basis"})
        return _report(T, basis, cv, n, m, total, rule)
    # Phase I: minimise the sum of the artificials.
    obj = [0.0] * (total + 1)
    for i in range(m):
        for k in range(total + 1):
            obj[k] -= T[i][k]
    for i in range(m):
        obj[n + i] = 0.0
    T.append(obj)
    st = _run(T, basis, range(n), rule, set(), max_iter)
    if st in ("cycling", "iteration_limit"):
        return _fail(st, rule, "phase 1")
    if -T[m][-1] > 1e-7:
        return RichResult(payload={
            "estimate": None, "status": "infeasible", "x": None,
            "fun": None, "message": "no point satisfies every "
                                    "constraint (phase 1 residual "
                                    "%.3g)" % (-T[m][-1]),
            "rule": rule,
            "method": "two-phase primal simplex (Dantzig 1963)"})
    # Drive any artificial still basic out of the basis if possible.
    for i in range(m):
        if basis[i] >= n:
            for j in range(n):
                if abs(T[i][j]) > _EPS:
                    _pivot(T, i, j)
                    basis[i] = j
                    break
    # Phase II: real objective, artificials blocked from re-entering.
    obj2 = [0.0] * (total + 1)
    for j in range(n):
        obj2[j] = cv[j]
    for i in range(m):
        if basis[i] < n and cv[basis[i]] != 0.0:
            f = cv[basis[i]]
            for k in range(total + 1):
                obj2[k] -= f * T[i][k]
    T[m] = obj2
    st = _run(T, basis, range(n), rule, set(range(n, total)),
              max_iter)
    if st in ("cycling", "iteration_limit"):
        return _fail(st, rule, "phase 2")
    if st == "unbounded":
        return RichResult(payload={
            "estimate": None, "status": "unbounded", "x": None,
            "fun": None, "rule": rule,
            "message": "the objective decreases without bound along "
                       "a feasible ray",
            "method": "two-phase primal simplex (Dantzig 1963)"})
    return _report(T, basis, cv, n, m, total, rule)


def _report(T, basis, cv, n, m, total, rule):
    x = [0.0] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i][-1]
    y = [-T[m][n + i] for i in range(m)]
    fun = sum(cv[j] * x[j] for j in range(n))
    degenerate = [basis[i] for i in range(m)
                  if abs(T[i][-1]) < _EPS]
    alt = [j for j in range(n)
           if j not in basis and abs(T[m][j]) < _EPS]
    return RichResult(payload={
        "estimate": x, "status": "optimal", "x": x, "fun": fun,
        "duals": y, "basis": list(basis),
        "reduced_costs": [T[m][j] for j in range(n)],
        "degenerate": degenerate,
        "multiple_optima": bool(alt), "alternate_entering": alt,
        "rule": rule,
        "method": "two-phase primal simplex (Dantzig 1963) with "
                  "%s's pivot rule" % ("Bland" if rule == "bland"
                                       else "Dantzig"),
    })


def _fail(st, rule, phase):
    why = ("the basis repeated, so the method is cycling"
           if st == "cycling" else "the iteration limit was reached")
    hint = (" -- Dantzig's rule can cycle on degenerate problems; "
            "rule='bland' is guaranteed to terminate"
            if rule == "dantzig" else "")
    return RichResult(payload={
        "estimate": None, "status": st, "x": None, "fun": None,
        "rule": rule,
        "message": "%s in %s%s" % (why, phase, hint),
        "method": "two-phase primal simplex (Dantzig 1963)"})


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
            upper=None, rule="bland", maximise=False,
            max_iter=10000):
    r"""Solve an inequality-form linear program.

    Minimises by default; ``maximise=True`` negates the objective and
    negates the reported value back.
    """
    sign = -1.0 if maximise else 1.0
    sf = standard_form([sign * float(v) for v in c], A_ub, b_ub,
                       A_eq, b_eq, upper)
    r = simplex(sf["c"], sf["A"], sf["b"], rule, max_iter)
    if r["status"] != "optimal":
        return r
    n = sf["n_original"]
    x = r["x"][:n]
    fun = sign * r["fun"]
    out = dict(r)
    out.update({"estimate": x, "x": x, "fun": fun,
                # Report duals in the sign convention of the problem
                # as posed, so b'y equals the reported objective.
                "duals": [sign * d for d in r["duals"]],
                "slack": r["x"][n:], "maximise": bool(maximise),
                "n_original": n, "n_slack": sf["n_slack"]})
    return RichResult(payload=out)
