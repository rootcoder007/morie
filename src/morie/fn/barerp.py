# morie.fn -- function file (rootcoder007/morie)
r"""The logarithmic barrier method for inequality-constrained convex problems.

Frisch, R. (1956) "La resolution des problemes de programme lineaire par
la methode du potentiel logarithmique", *Cahiers du Seminaire
d'Econometrie* No. 4, 7-23. JSTOR 20075373.

Frisch, R. (1957) "The Multiplex Method for Linear Programming",
*Sankhya* 18(3/4), 329-362. JSTOR 25048355.

Boyd, S., & Vandenberghe, L. (2004) *Convex Optimization*, Cambridge
University Press, Chapter 11.

Frisch's idea, and his own words for it: "we work systematically in the
interior of the admissible region and use a logarithmic potential as a
guide -- a sort of radar -- to keep us from crossing the boundary." His
potential, equation 5.1, is

.. math::   V = \sum_k \log x_k + \sum_j \log X_j,

"simply the sum of the logarithms of all the variables, the basic
variables as well as the dependent ones" -- the slacks included. It is
smooth in the interior and falls to :math:`-\infty` at every boundary
point. He then moves along a compromise between the preference gradient
:math:`p` and the potential gradient :math:`\nabla V`: "to increase the
preference function we must go in the direction :math:`p`, but to keep
away from the boundary we must go in the direction :math:`V_k`."

That compromise is the modern central path. Writing the problem as

.. math::

   \text{minimize } f_0(x) \text{ subject to }
   f_i(x) \leq 0,\; i = 1,\ldots,m, \quad Ax = b,

Boyd's Chapter 11 defines the log barrier (equation 11.5)

.. math::   \phi(x) = -\sum_{i=1}^{m} \log(-f_i(x)),

which is :math:`-V` with :math:`-f_i` the slacks, and follows
:math:`x^{\star}(t) = \arg\min\, t f_0(x) + \phi(x)`. Frisch's weighted
compromise of the two gradients is a step on exactly this function; the
only thing Boyd adds is to take that step by Newton's method instead.

The barrier's derivatives (Boyd p. 564):

.. math::

   \nabla \phi(x) = \sum_i \frac{1}{-f_i(x)} \nabla f_i(x), \qquad
   \nabla^2 \phi(x) = \sum_i \frac{1}{f_i(x)^2}
        \nabla f_i(x) \nabla f_i(x)^T
     + \sum_i \frac{1}{-f_i(x)} \nabla^2 f_i(x).

What makes the method finish is that the central point comes with its
own certificate. From :math:`x^{\star}(t)`,

.. math::

   \lambda_i^{\star}(t) = \frac{-1}{t f_i(x^{\star}(t))}, \qquad
   \nu^{\star}(t) = \nu / t

is dual feasible, and the duality gap is exactly :math:`m/t` -- one
number, independent of the problem. So :math:`m/t < \epsilon` is a
guarantee, not a heuristic.

**Algorithm 11.1.** Given strictly feasible :math:`x`,
:math:`t := t^{(0)} > 0`, :math:`\mu > 1`, :math:`\epsilon > 0`:
repeat -- centre (compute :math:`x^{\star}(t)` starting from :math:`x`),
update :math:`x := x^{\star}(t)`, quit if :math:`m/t < \epsilon`,
otherwise :math:`t := \mu t`. Equation 11.13 pins the count of centering
steps at :math:`\lceil \log(m/(\epsilon t^{(0)})) / \log \mu \rceil`
beyond the first.

Three centering routes are kept, since the sources give three.

``centering="newton"`` is Boyd's, and the default: Newton's method with
a backtracking line search, stopping on the Newton decrement. It is the
one to use.

``centering="gradient"`` is Frisch's own compromise of the two
gradients, taken as a step on :math:`t f_0 + \phi`. It reaches the same
central path and wants far more steps, which is the reason Newton
displaced it.

``centering="none"`` is Boyd's opening remark in section 11.3: set
:math:`t = m/\epsilon` once and centre a single time. He notes it "can
work well for small problems, good starting points, and moderate
accuracy" but "is rarely, if ever, used". It is here because it is the
baseline the barrier method is an improvement on.

A strictly feasible starting point can be found with :func:`phase1`
(Boyd section 11.4: minimise :math:`s` subject to
:math:`f_i(x) \leq s`), which is the role of Frisch's own section VI.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "barerp",
    "barrier_method",
    "barriermethod",
    "barrier_lp",
    "phase1",
    "log_barrier",
    "log_barrier_gradient",
    "log_barrier_hessian",
    "frisch_potential",
    "central_point",
    "central_path_dual",
    "centering_steps",
]

_CENTERING = ("newton", "gradient", "none")


# --------------------------------------------------------------------------
# the barrier itself
# --------------------------------------------------------------------------

def log_barrier(fvals):
    r"""Boyd equation 11.5: :math:`\phi(x) = -\sum_i \log(-f_i(x))`.

    Infinite outside the strict interior, which is the whole point.
    """
    out = 0.0
    for v in fvals:
        v = float(v)
        if v >= 0.0:
            return float("inf")
        out -= math.log(-v)
    return out


def frisch_potential(slacks):
    r"""Frisch equation 5.1: :math:`V = \sum_k \log x_k`.

    The sum of the logarithms of all the variables. With the slacks
    :math:`-f_i(x)` as the variables this is :math:`-\phi(x)`.
    """
    out = 0.0
    for v in slacks:
        v = float(v)
        if v <= 0.0:
            return float("-inf")
        out += math.log(v)
    return out


def log_barrier_gradient(fvals, jac):
    r""":math:`\nabla \phi = \sum_i \nabla f_i / (-f_i)` (Boyd p. 564)."""
    n = len(jac[0]) if jac else 0
    out = [0.0] * n
    for i, v in enumerate(fvals):
        w = 1.0 / (-float(v))
        gi = jac[i]
        for j in range(n):
            out[j] += w * gi[j]
    return out


def log_barrier_hessian(fvals, jac, hess=None):
    r""":math:`\nabla^2 \phi = \sum_i \nabla f_i \nabla f_i^T / f_i^2
    + \sum_i \nabla^2 f_i / (-f_i)` (Boyd p. 564).

    ``hess`` may be omitted when every constraint is affine, since then
    the second term vanishes.
    """
    n = len(jac[0]) if jac else 0
    out = [[0.0] * n for _ in range(n)]
    for i, v in enumerate(fvals):
        v = float(v)
        gi = jac[i]
        w = 1.0 / (v * v)
        for a in range(n):
            if gi[a] == 0.0:
                continue
            for b in range(n):
                out[a][b] += w * gi[a] * gi[b]
        if hess is not None and hess[i] is not None:
            w2 = 1.0 / (-v)
            hi = hess[i]
            for a in range(n):
                for b in range(n):
                    out[a][b] += w2 * float(hi[a][b])
    return out


def central_path_dual(fvals, t):
    r"""Boyd equation 11.10: :math:`\lambda_i^{\star}(t) = -1/(t f_i(x))`.

    Strictly positive at any strictly feasible point, and the pair
    :math:`(\lambda^{\star}, \nu^{\star})` it belongs to certifies a
    duality gap of exactly :math:`m/t`.
    """
    t = float(t)
    if t <= 0.0:
        raise ValueError("barerp: t must be positive")
    return [-1.0 / (t * float(v)) for v in fvals]


def centering_steps(m, eps, t0, mu):
    r"""Boyd equation 11.13: the exact number of centering steps.

    :math:`\lceil \log(m/(\epsilon t^{(0)})) / \log \mu \rceil` beyond
    the initial one, and never negative.
    """
    if mu <= 1.0:
        raise ValueError("barerp: mu must exceed 1")
    if eps <= 0.0 or t0 <= 0.0:
        raise ValueError("barerp: eps and t0 must be positive")
    val = math.log(m / (eps * t0)) / math.log(mu)
    return max(0, int(math.ceil(val - 1e-12)))


# --------------------------------------------------------------------------
# derivatives, supplied or differenced
# --------------------------------------------------------------------------

def _num_grad(f, x, h=1e-6):
    out = []
    for j in range(len(x)):
        step = h * max(1.0, abs(x[j]))
        up = list(x)
        dn = list(x)
        up[j] += step
        dn[j] -= step
        out.append((f(up) - f(dn)) / (2.0 * step))
    return out


def _num_hess(f, x, h=1e-4):
    n = len(x)
    out = [[0.0] * n for _ in range(n)]
    f0 = f(x)
    for a in range(n):
        sa = h * max(1.0, abs(x[a]))
        for b in range(a, n):
            sb = h * max(1.0, abs(x[b]))
            xpp = list(x)
            xpm = list(x)
            xmp = list(x)
            xmm = list(x)
            xpp[a] += sa
            xpp[b] += sb
            xpm[a] += sa
            xpm[b] -= sb
            xmp[a] -= sa
            xmp[b] += sb
            xmm[a] -= sa
            xmm[b] -= sb
            if a == b:
                val = (f(xpp) - 2.0 * f0 + f(xmm)) / (4.0 * sa * sa)
            else:
                val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * sa * sb)
            out[a][b] = val
            out[b][a] = val
    return out


class _Fun(object):
    """A function with whatever derivatives were supplied."""

    def __init__(self, f, grad=None, hess=None, affine=False):
        self.f = f
        self._g = grad
        self._h = hess
        self.affine = affine

    def val(self, x):
        return float(self.f(x))

    def grad(self, x):
        if self._g is not None:
            return [float(v) for v in self._g(x)]
        return _num_grad(self.f, x)

    def hess(self, x):
        if self._h is not None:
            return [[float(v) for v in row] for row in self._h(x)]
        if self.affine:
            n = len(x)
            return [[0.0] * n for _ in range(n)]
        return _num_hess(self.f, x)


def _as_fun(spec):
    if isinstance(spec, _Fun):
        return spec
    if callable(spec):
        return _Fun(spec)
    if isinstance(spec, dict):
        return _Fun(spec["f"], spec.get("grad"), spec.get("hess"),
                    bool(spec.get("affine", False)))
    raise ValueError("barerp: a constraint must be a callable or a dict "
                     "with 'f' and optionally 'grad', 'hess', 'affine'")


# --------------------------------------------------------------------------
# the KKT solve for one Newton step
# --------------------------------------------------------------------------

def _solve_kkt(hmat, grad, aeq):
    """Boyd equation 11.14: the Newton step with equality constraints."""
    n = len(grad)
    if not aeq:
        return [float(v) for v in
                np.linalg.solve(np.asarray(hmat, dtype=float),
                                np.asarray([-g for g in grad], dtype=float))]
    p = len(aeq)
    big = [[0.0] * (n + p) for _ in range(n + p)]
    for a in range(n):
        for b in range(n):
            big[a][b] = hmat[a][b]
    for r in range(p):
        for c in range(n):
            big[n + r][c] = float(aeq[r][c])
            big[c][n + r] = float(aeq[r][c])
    rhs = [-g for g in grad] + [0.0] * p
    sol = np.linalg.solve(np.asarray(big, dtype=float),
                          np.asarray(rhs, dtype=float))
    return [float(sol[j]) for j in range(n)]


# --------------------------------------------------------------------------
# centering
# --------------------------------------------------------------------------

def central_point(f0, cons, x, t, aeq=None, centering="newton",
                  tol=1e-10, max_iter=200, alpha=0.01, beta=0.5,
                  step0=1.0):
    r"""Compute :math:`x^{\star}(t) = \arg\min t f_0 + \phi`.

    ``centering="newton"`` uses Newton's method with a backtracking line
    search and stops when the Newton decrement satisfies
    :math:`\lambda^2/2 \leq` ``tol``; ``"gradient"`` takes Frisch's
    compromise of the preference and potential gradients as a plain
    descent step on the same function.
    """
    x = [float(v) for v in x]
    n = len(x)
    t = float(t)

    def objective(z):
        fv = [c.val(z) for c in cons]
        for v in fv:
            if v >= 0.0:
                return float("inf")
        return t * f0.val(z) + log_barrier(fv)

    cur = objective(x)
    if cur == float("inf"):
        raise ValueError("barerp: the starting point is not strictly "
                         "feasible")

    iters = 0
    decrement = float("inf")
    for iters in range(1, max_iter + 1):
        fv = [c.val(x) for c in cons]
        jac = [c.grad(x) for c in cons]
        g0 = f0.grad(x)
        gphi = log_barrier_gradient(fv, jac)
        grad = [t * g0[j] + gphi[j] for j in range(n)]

        if centering == "newton":
            hs = None
            if any(not c.affine for c in cons):
                hs = [None if c.affine else c.hess(x) for c in cons]
            hmat = log_barrier_hessian(fv, jac, hs)
            h0 = f0.hess(x)
            for a in range(n):
                for b in range(n):
                    hmat[a][b] += t * float(h0[a][b])
            try:
                step = _solve_kkt(hmat, grad, aeq)
            except Exception:
                break
            decrement = -sum(grad[j] * step[j] for j in range(n))
            if decrement / 2.0 <= tol:
                break
        else:
            # Frisch: move along a compromise of the preference gradient
            # and the potential gradient, which together are the gradient
            # of t f_0 + phi.
            step = [-g for g in grad]
            if aeq:
                step = _project_null(step, aeq)
            decrement = sum(g * g for g in grad)
            if decrement <= tol:
                break

        s = float(step0)
        gts = sum(grad[j] * step[j] for j in range(n))
        for _ in range(80):
            trial = [x[j] + s * step[j] for j in range(n)]
            val = objective(trial)
            if val <= cur + alpha * s * gts:
                break
            s *= beta
        else:
            break
        x = [x[j] + s * step[j] for j in range(n)]
        cur = val

    return x, iters, decrement


def _project_null(v, aeq):
    """Project onto the null space of A, so Ax = b is preserved."""
    am = np.asarray([[float(c) for c in row] for row in aeq], dtype=float)
    vv = np.asarray([float(c) for c in v], dtype=float)
    gram = np.dot(am, np.asarray(am).T)
    rhs = np.dot(am, vv)
    lam = np.linalg.solve(np.asarray(gram), np.asarray(rhs))
    corr = np.dot(np.asarray(am).T, np.asarray(lam))
    return [float(vv[j]) - float(corr[j]) for j in range(len(v))]


# --------------------------------------------------------------------------
# phase I
# --------------------------------------------------------------------------

def phase1(cons, x0, aeq=None, beq=None, max_outer=60, **kw):
    r"""Boyd section 11.4: minimise :math:`s` subject to
    :math:`f_i(x) \leq s`.

    Any :math:`x` works as a start, since :math:`s` can be set above
    every :math:`f_i(x)`. A strictly feasible point for the original
    problem exists exactly when the optimal :math:`s` is negative --
    which is Frisch's section VI, "the search for a point in the
    admissible region", done by the same machinery.
    """
    cons = [_as_fun(c) for c in cons]
    x0 = [float(v) for v in x0]
    n = len(x0)
    fv = [c.val(x0) for c in cons]
    s0 = max(fv) + 1.0

    def lift(c):
        return _Fun(lambda z, _c=c: _c.val(z[:n]) - z[n],
                    lambda z, _c=c: list(_c.grad(z[:n])) + [-1.0],
                    None, affine=c.affine)

    lifted = [lift(c) for c in cons]
    obj = _Fun(lambda z: z[n],
               lambda z: [0.0] * n + [1.0], None, affine=True)
    aeq2 = None
    if aeq:
        aeq2 = [[float(v) for v in row] + [0.0] for row in aeq]

    res = barrier_method(obj, lifted, list(x0) + [s0], aeq=aeq2, beq=beq,
                         max_outer=max_outer, **kw)
    z = res["x"]
    s = z[n]
    return {"x": z[:n], "s": float(s), "feasible": bool(s < 0.0),
            "outer": res["outer"], "newton": res["newton"]}


# --------------------------------------------------------------------------
# Algorithm 11.1
# --------------------------------------------------------------------------

def barrier_method(f0, constraints, x0, t0=1.0, mu=10.0, eps=1e-8,
                   aeq=None, beq=None, centering="newton", tol=1e-10,
                   max_inner=200, max_outer=200, grad=None, hess=None,
                   affine=False):
    r"""Boyd Algorithm 11.1, the barrier (path-following) method.

    Parameters
    ----------
    f0 : callable or dict
        The objective. A dict may carry ``grad``, ``hess`` and
        ``affine``; anything absent is differenced.
    constraints : sequence
        The :math:`f_i` with :math:`f_i(x) \leq 0`, in the same form.
    x0 : sequence
        A **strictly** feasible starting point -- :func:`phase1` will
        find one. Frisch: start "from a point in the interior of the
        admissible region".
    t0, mu, eps :
        :math:`t^{(0)}`, the factor :math:`\mu > 1` by which :math:`t`
        grows each outer iteration, and the duality-gap tolerance.
        Boyd: :math:`\mu` from 10 to 20 works well, and the method is
        insensitive over roughly 3 to 100.
    aeq, beq :
        Equality constraints :math:`Ax = b`, kept exactly by the Newton
        step of equation 11.14. ``x0`` must already satisfy them.
    centering : {"newton", "gradient", "none"}
        See the module docstring: Boyd's Newton centering, Frisch's
        gradient compromise, or the single-shot :math:`t = m/\epsilon`.

    Returns
    -------
    RichResult
        ``x``, ``fun``, ``gap`` (:math:`m/t`, an actual bound),
        ``lambda_`` (the equation 11.10 dual point), ``t``, ``outer``
        and ``newton`` counts, and the per-outer-iteration ``history``.
    """
    if centering not in _CENTERING:
        raise ValueError("barerp: centering must be one of %s"
                         % (_CENTERING,))
    if mu <= 1.0:
        raise ValueError("barerp: mu must exceed 1")
    if t0 <= 0.0 or eps <= 0.0:
        raise ValueError("barerp: t0 and eps must be positive")

    f0 = _as_fun(f0 if not callable(f0) or grad is None else
                 {"f": f0, "grad": grad, "hess": hess, "affine": affine})
    cons = [_as_fun(c) for c in constraints]
    m = len(cons)
    if m == 0:
        raise ValueError("barerp: no inequality constraints; this is an "
                         "unconstrained problem")
    x = [float(v) for v in x0]
    if any(c.val(x) >= 0.0 for c in cons):
        raise ValueError("barerp: x0 is not strictly feasible; use "
                         "phase1() to find a starting point")
    if aeq:
        aeq = [[float(v) for v in row] for row in aeq]
        if beq is not None:
            for r, row in enumerate(aeq):
                lhs = sum(row[j] * x[j] for j in range(len(x)))
                if abs(lhs - float(beq[r])) > 1e-8:
                    raise ValueError(
                        "barerp: x0 violates equality row %d by %g"
                        % (r, lhs - float(beq[r])))

    if centering == "none":
        t = m / eps
        x, it, dec = central_point(f0, cons, x, t, aeq, "newton", tol,
                                   max_inner)
        fv = [c.val(x) for c in cons]
        return RichResult(payload={
            "x": x, "fun": f0.val(x), "gap": m / t, "t": t,
            "lambda_": central_path_dual(fv, t), "slack": [-v for v in fv],
            "outer": 1, "newton": it, "decrement": dec,
            "history": [(t, m / t, f0.val(x), it)],
            "centering": "none", "converged": True,
            "method": ("single centering at t = m/eps, Boyd sec. 11.3 "
                       "opening -- 'rarely, if ever, used'"),
        })

    t = float(t0)
    total = 0
    history = []
    outer = 0
    converged = False
    for outer in range(1, max_outer + 1):
        x, it, dec = central_point(f0, cons, x, t, aeq, centering, tol,
                                   max_inner)
        total += it
        history.append((t, m / t, f0.val(x), it))
        if m / t < eps:
            converged = True
            break
        t *= mu

    fv = [c.val(x) for c in cons]
    return RichResult(payload={
        "x": x,
        "fun": f0.val(x),
        "gap": m / t,
        "t": t,
        "lambda_": central_path_dual(fv, t),
        "slack": [-v for v in fv],
        "outer": outer,
        "newton": total,
        "decrement": dec,
        "history": history,
        "centering": centering,
        "converged": converged,
        "steps_predicted": centering_steps(m, eps, float(t0), float(mu)),
        "method": ("Boyd Algorithm 11.1, the logarithmic barrier method "
                   "of Frisch (1956) eq. 5.1"),
        "note": ("gap is m/t, the exact duality gap certified by the "
                 "central point's dual pair (Boyd eq. 11.10-11.12), not "
                 "an estimate"),
    })


def barrier_lp(c, A_ub, b_ub, A_eq=None, b_eq=None, x0=None, **kw):
    r"""The barrier method on a linear program.

    minimise :math:`c^T x` subject to :math:`A_{ub} x \leq b_{ub}` and
    optionally :math:`A_{eq} x = b_{eq}`. With no ``x0`` a strictly
    feasible point is found by :func:`phase1` first.
    """
    c = [float(v) for v in c]
    n = len(c)
    rows = [[float(v) for v in row] for row in A_ub]
    b = [float(v) for v in b_ub]
    if len(rows) != len(b):
        raise ValueError("barerp: A_ub has %d rows but b_ub has %d"
                         % (len(rows), len(b)))
    for row in rows:
        if len(row) != n:
            raise ValueError("barerp: A_ub row width does not match c")

    obj = {"f": lambda z: sum(c[j] * z[j] for j in range(n)),
           "grad": lambda z: list(c), "affine": True}
    cons = [{"f": (lambda z, _r=row, _bi=bi:
                   sum(_r[j] * z[j] for j in range(n)) - _bi),
             "grad": (lambda z, _r=row: list(_r)),
             "affine": True}
            for row, bi in zip(rows, b)]

    if x0 is None:
        ph = phase1(cons, [0.0] * n, aeq=A_eq, beq=b_eq)
        if not ph["feasible"]:
            raise ValueError("barerp: no strictly feasible point found; "
                             "phase1 stopped at s = %g" % ph["s"])
        x0 = ph["x"]
    res = barrier_method(obj, cons, x0, aeq=A_eq, beq=b_eq, **kw)
    return res


barerp = barrier_method
barriermethod = barrier_method


def cheatsheet():
    return ("barerp: the logarithmic barrier method. Frisch (1956) "
            "eq. 5.1 defines the potential as the sum of the logs of "
            "all the variables -- slacks included -- and moves along a "
            "compromise between the preference gradient and the "
            "potential gradient, staying inside the admissible region. "
            "Boyd ch. 11 is the same path: minimise t f0 + phi with "
            "phi = -sum log(-f_i), for t growing by mu each outer "
            "iteration (Algorithm 11.1). The central point carries its "
            "own certificate -- lambda_i = -1/(t f_i) is dual feasible "
            "and the duality gap is exactly m/t -- so m/t < eps is a "
            "guarantee. centering='newton' (default), 'gradient' "
            "(Frisch's own), or 'none' (single shot at t = m/eps). "
            "phase1() finds a strictly feasible start.")
