# morie.fn -- function file (rootcoder007/morie)
"""Courant's quadratic penalty method for constrained minimisation.

Source FETCHED and read: Courant, R. (1943), "Variational methods for
the solution of problems of equilibrium and vibrations", *Bulletin of
the American Mathematical Society* 49(1):1-23, open access at the AMS
(S0002-9904-1943-07818-4).  Section 2b, "Rigid constraints as limiting
cases", page 8, is the origin of the method.  Courant adds to the
Dirichlet functional a boundary term K(v) = integral_C v^2 ds weighted
by a parameter,

    Q(v) = D(v) + gamma K(v),

and observes that "as the parameter gamma increases indefinitely ...
the condition (10) tends to the boundary condition v = 0 of the clamped
membrane", concluding that "quite generally rigid boundary conditions
should be regarded as limiting cases of natural conditions in which a
parameter tends to infinity".  That is exactly the penalty device: the
hard constraint is recovered in the limit of an unboundedly weighted
squared violation.  In modern notation, for constraints g_i(x) <= 0,

    Q(x; mu) = f(x) + mu * sum_i max(0, g_i(x))^2

is minimised for an increasing sequence of mu, each subproblem started
from the previous solution.

The inner minimiser is steepest descent with an Armijo backtracking line
search, using central finite differences for the gradient.  It runs a
FIXED number of steps and halves the trial step a FIXED maximum number
of times, because an early exit taken on one language arm and not the
other would silently break Python/R parity.
"""

from ._richresult import RichResult

__all__ = ["penalty_method"]


def _violation(constraints, x):
    return [max(0.0, float(g(x))) for g in constraints]


def _q(f, constraints, x, mu):
    v = _violation(constraints, x)
    return float(f(x)) + mu * sum(vi * vi for vi in v)


def _grad(fun, x, h):
    g = []
    for k in range(len(x)):
        xp = list(x)
        xm = list(x)
        xp[k] += h
        xm[k] -= h
        g.append((fun(xp) - fun(xm)) / (2.0 * h))
    return g


def penalty_method(f, constraints, x0, mu, n_outer=8, growth=10.0,
                   n_inner=200, step0=1.0, h=1e-6, armijo=1e-4,
                   max_halving=40):
    """Minimise f subject to g_i(x) <= 0 by Courant's quadratic penalty.

    Parameters
    ----------
    f : callable
        Objective, taking a list of floats and returning a float.
    constraints : sequence of callables
        Each returns g_i(x); the constraint holds when g_i(x) <= 0.  An
        equality h(x) = 0 is expressed as ``lambda x: abs(h(x))``, whose
        square is h(x)^2.
    x0 : sequence of float
        Starting point.
    mu : float
        Initial penalty weight, multiplied by ``growth`` each outer pass.
    n_outer, growth, n_inner, step0, h, armijo, max_halving
        Fixed-budget controls of the outer penalty loop and the inner
        steepest-descent solve.

    Returns
    -------
    RichResult
        Keys ``x``, ``f``, ``penalty``, ``violation``, ``max_violation``,
        ``q``, ``mu``, ``n_outer``, ``n_inner``, ``method``.
    """
    x = [float(v) for v in x0]
    constraints = list(constraints)
    mu = float(mu)
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if growth <= 1.0:
        raise ValueError("growth must exceed 1 or the penalty never tightens")
    for _outer in range(int(n_outer)):
        def q(z, _mu=mu):
            return _q(f, constraints, z, _mu)
        cur = q(x)
        for _inner in range(int(n_inner)):
            g = _grad(q, x, h)
            gn2 = sum(v * v for v in g)
            if gn2 == 0.0:
                break
            step = float(step0)
            improved = False
            for _hv in range(int(max_halving)):
                trial = [x[k] - step * g[k] for k in range(len(x))]
                qt = q(trial)
                if qt <= cur - armijo * step * gn2:
                    x = trial
                    cur = qt
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
        mu *= growth
    v = _violation(constraints, x)
    fx = float(f(x))
    pen = sum(vi * vi for vi in v)
    return RichResult(
        payload={
            "x": x,
            "f": fx,
            "penalty": pen,
            "violation": v,
            "max_violation": max(v) if v else 0.0,
            "q": fx + (mu / growth) * pen,
            "mu": mu / growth,
            "n_outer": int(n_outer),
            "n_inner": int(n_inner),
            "method": "Courant (1943) quadratic penalty method",
        }
    )


def cheatsheet():
    return "penmth: Penalty method for constrained"


# compact alias per ledger/NAMING.md
penaltymethod = penalty_method
