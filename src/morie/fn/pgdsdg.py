# morie.fn -- function file (rootcoder007/morie)
r"""Projected gradient descent, and the projections that make it work.

**The method.** To minimise :math:`f` over a closed convex set
:math:`C`, take a gradient step and put the result back:

.. math:: x_{k+1} = P_C\bigl(x_k - t_k \nabla f(x_k)\bigr).

Goldstein's observation is that this is not a heuristic repair. The
projection :math:`P_C` onto a closed convex set is *non-expansive*,
:math:`\|P_C u - P_C v\| \le \|u - v\|`, so it cannot undo the
contraction the gradient step achieves. For :math:`f` convex with
:math:`L`-Lipschitz gradient and :math:`t \le 1/L`, the iterates
converge at rate :math:`O(1/k)`.

**The fixed point is the optimality condition.** :math:`x^*` solves
the constrained problem exactly when :math:`x^* = P_C(x^* - t\nabla
f(x^*))` for any :math:`t > 0` -- which is the projected form of the
KKT conditions, and is what the anchor checks rather than merely
watching the objective stop moving.

**Two accelerations, and one honest limitation.**

``backtracking`` searches for a step satisfying the descent
inequality, which removes the need to know :math:`L`.

``fista`` adds Nesterov momentum with the Beck-Teboulle
:math:`t_{k+1} = (1 + \sqrt{1 + 4t_k^2})/2` sequence, improving the
rate to :math:`O(1/k^2)`. It is *not* monotone: the objective can
rise on individual iterations, and an implementation that stops on
"objective increased" will stop early. The objective history is
returned so this is visible rather than surprising.

The limitation: all of this assumes :math:`f` convex. On a non-convex
:math:`f` the method still converges to a stationary point of the
constrained problem, and nothing here says it is a minimum.

**Projections included.** Box, non-negative orthant, Euclidean ball,
and the probability simplex. The simplex projection is the
interesting one -- it is not clipping-and-renormalising, which is a
different and wrong operation. The correct projection solves for a
threshold :math:`\theta` with :math:`\sum_i \max(x_i - \theta, 0) =
1`, and the anchor checks it against a brute-force minimisation
rather than against the formula it was implemented from.

References
----------
Goldstein, A. A. (1964) "Convex programming in Hilbert space",
*Bulletin of the American Mathematical Society* 70(5), 709-710,
doi:10.1090/S0002-9904-1964-11178-2. The projected gradient
iteration above and the non-expansiveness of the projection onto a
closed convex set.

Levitin, E. S. & Polyak, B. T. (1966) "Constrained minimization
methods", *USSR Computational Mathematics and Mathematical Physics*
6(5), 1-50, doi:10.1016/0041-5553(66)90114-5. The convergence
analysis for the constrained gradient method and the step-size
condition :math:`t \le 1/L`.

Beck, A. & Teboulle, M. (2009) "A fast iterative
shrinkage-thresholding algorithm for linear inverse problems", *SIAM
Journal on Imaging Sciences* 2(1), 183-202, doi:10.1137/080716542.
The momentum sequence and the :math:`O(1/k^2)` rate used by
``fista``, and the backtracking rule.
"""

import math

from ._richresult import RichResult

__all__ = ["project_box", "project_nonneg", "project_ball",
           "project_simplex", "projected_gradient", "STEP_RULES",
           "projected_gradient_descent"]

STEP_RULES = ("fixed", "backtracking", "fista")


def project_box(x, lower=None, upper=None):
    r"""Clip into :math:`[l, u]`, coordinatewise."""
    v = [float(t) for t in x]
    n = len(v)
    lo = [float("-inf")] * n if lower is None else [
        float("-inf") if t is None else float(t) for t in lower]
    up = [float("inf")] * n if upper is None else [
        float("inf") if t is None else float(t) for t in upper]
    if len(lo) != n or len(up) != n:
        raise ValueError("pgdsdg: the bounds must have one entry per "
                         "coordinate (%d)" % n)
    if any(lo[i] > up[i] for i in range(n)):
        raise ValueError("pgdsdg: a lower bound exceeds its upper "
                         "bound, so the box is empty")
    return [min(max(v[i], lo[i]), up[i]) for i in range(n)]


def project_nonneg(x):
    r"""Projection onto :math:`\{x \ge 0\}`."""
    return [max(0.0, float(t)) for t in x]


def project_ball(x, radius=1.0, centre=None):
    r"""Projection onto a Euclidean ball."""
    r = float(radius)
    if r <= 0:
        raise ValueError("pgdsdg: the ball radius must be positive")
    v = [float(t) for t in x]
    c = [0.0] * len(v) if centre is None else [float(t)
                                               for t in centre]
    if len(c) != len(v):
        raise ValueError("pgdsdg: the centre has %d coordinates but "
                         "the point has %d" % (len(c), len(v)))
    d = [v[i] - c[i] for i in range(len(v))]
    nrm = math.sqrt(sum(t * t for t in d))
    if nrm <= r:
        return v
    return [c[i] + d[i] * r / nrm for i in range(len(v))]


def project_simplex(x, total=1.0):
    r"""Projection onto :math:`\{x \ge 0,\ \sum x_i = s\}`.

    Solves for the threshold :math:`\theta` with
    :math:`\sum_i \max(x_i - \theta, 0) = s`. This is *not* clipping
    at zero and renormalising, which lands somewhere else entirely.
    """
    s = float(total)
    if s <= 0:
        raise ValueError("pgdsdg: the simplex total must be positive")
    v = [float(t) for t in x]
    n = len(v)
    if n == 0:
        raise ValueError("pgdsdg: cannot project an empty vector")
    u = sorted(v, reverse=True)
    css = 0.0
    rho, theta = 0, (u[0] - s)
    for i in range(n):
        css += u[i]
        t = (css - s) / (i + 1)
        if u[i] - t > 0:
            rho = i + 1
            theta = t
    return [max(v[i] - theta, 0.0) for i in range(n)]


def _norm(v):
    return math.sqrt(sum(t * t for t in v))


def projected_gradient(f, grad, x0, project, step=None,
                       rule="backtracking", max_iter=2000,
                       tol=1e-10):
    r"""Minimise ``f`` over the set ``project`` projects onto.

    ``project`` is any callable implementing :math:`P_C`; the four
    above are provided, and any non-expansive projection works.
    """
    if rule not in STEP_RULES:
        raise ValueError("pgdsdg: rule must be one of %s, got %r"
                         % (", ".join(STEP_RULES), rule))
    x = project([float(v) for v in x0])
    n = len(x)
    t = 1.0 if step is None else float(step)
    if t <= 0:
        raise ValueError("pgdsdg: the step size must be positive")
    if rule == "fixed" and step is None:
        raise ValueError("pgdsdg: a fixed step rule needs an explicit "
                         "step size")
    hist = [float(f(x))]
    y, tk = list(x), 1.0
    n_back = 0
    for it in range(int(max_iter)):
        base = y if rule == "fista" else x
        g = [float(v) for v in grad(base)]
        if len(g) != n:
            raise ValueError("pgdsdg: the gradient has %d components "
                             "but the point has %d" % (len(g), n))
        if rule == "fixed":
            new = project([base[i] - t * g[i] for i in range(n)])
        else:
            fb = float(f(base))
            for _ in range(60):
                cand = project([base[i] - t * g[i] for i in range(n)])
                d = [cand[i] - base[i] for i in range(n)]
                q = (fb + sum(g[i] * d[i] for i in range(n))
                     + sum(v * v for v in d) / (2.0 * t))
                if float(f(cand)) <= q + 1e-12:
                    break
                t *= 0.5
                n_back += 1
            else:
                raise ValueError("pgdsdg: backtracking failed to find "
                                 "a step satisfying the descent "
                                 "inequality; is the gradient correct?")
            new = cand
        if rule == "fista":
            tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
            y = [new[i] + ((tk - 1.0) / tn) * (new[i] - x[i])
                 for i in range(n)]
            tk = tn
        move = _norm([new[i] - x[i] for i in range(n)])
        x = new
        hist.append(float(f(x)))
        if move < tol:
            break
    g = [float(v) for v in grad(x)]
    fixed = project([x[i] - 1.0 * g[i] for i in range(n)])
    resid = _norm([fixed[i] - x[i] for i in range(n)])
    return RichResult(payload={
        "estimate": x, "x": x, "fun": float(f(x)),
        "iterations": it + 1, "history": hist,
        "step": t, "rule": rule, "n_backtracks": n_back,
        "fixed_point_residual": resid,
        "converged": resid < 1e-6,
        "monotone": all(hist[i] >= hist[i + 1] - 1e-12
                        for i in range(len(hist) - 1)),
        "method": "projected gradient (Goldstein 1964; "
                  "Levitin & Polyak 1966)"
                  + (" with Beck-Teboulle momentum"
                     if rule == "fista" else ""),
    })


def projected_gradient_descent(f, grad, x0, project, **kw):
    r"""Entry point: see :func:`projected_gradient`."""
    return projected_gradient(f, grad, x0, project, **kw)
