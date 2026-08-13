# morie.fn -- function file (rootcoder007/morie)
r"""Nonlinear conjugate gradients.

Fletcher, R., & Reeves, C. M. (1964) "Function minimization by conjugate
gradients", *The Computer Journal* 7(2), 149-154.
doi:10.1093/comjnl/7.2.149

Polak, E., & Ribiere, G. (1969) "Note sur la convergence de methodes de
directions conjuguees", *Revue francaise d'informatique et de recherche
operationnelle, Serie rouge* 3(R1), 35-43. Open access at Numdam,
http://www.numdam.org/article/M2AN_1969__3_1_35_0.pdf -- section 3.2
and equation 3.20, for the Polak-Ribiere choice of :math:`\beta`.

Shewchuk, J. R. (1994) "An Introduction to the Conjugate Gradient Method
Without the Agonizing Pain", Edition 1 1/4, School of Computer Science,
Carnegie Mellon University, CMU-CS-94-125 -- section 14.1, for the
``max(beta, 0)`` safeguard only, which is not in Polak & Ribiere.

Fletcher & Reeves is the primary source and everything below is theirs
unless said otherwise.

The appeal is stated in the abstract: a quadratically convergent
gradient method whose "particular advantages are its simplicity and its
modest demands on storage, space for only three vectors being
required". That is the contrast with Davidon-Fletcher-Powell, which
carries a full :math:`n \times n` matrix :math:`H_i` and so buys
curvature information at the price of storing it.

**The algorithm, equation 20.**

.. math::

   x_0 &= \text{arbitrary} \\
   g_0 &= g(x_0), \quad p_0 = -g_0 \\
   x_{i+1} &= \text{the minimum of } f \text{ along the line through }
              x_i \text{ in the direction } p_i \\
   g_{i+1} &= g(x_{i+1}) \\
   \beta_i &= \frac{g_{i+1}' g_{i+1}}{g_i' g_i} \\
   p_{i+1} &= -g_{i+1} + \beta_i p_i

"This process is guaranteed, apart from rounding errors, to locate the
minimum of any quadratic function of :math:`n` arguments in at most
:math:`n` iterations." That is a sharp claim and the tests check it
directly, along with the conjugacy :math:`p_i' A p_j = 0` and the
gradient orthogonality :math:`g_i' g_j = 0` that produce it.

**Restarts.** On Rosenbrock's banana-shaped valley the authors found
successive directions "so nearly parallel that the points :math:`x_i`
were scarcely separated", and the path "swung wide on the bend". Their
fix is to revert periodically to steepest descent, "discarding all
previous experience, whether useful or erroneous". Quadratic
convergence survives "provided that such restarts are not more frequent
than every :math:`n` iterations"; in practice they restart every
:math:`n+1`, which is the default here.

**The line search.** Theirs is in three stages. A unit of :math:`t` is
chosen to correspond to a displacement along :math:`p_i` of unit length
in :math:`x`-space, supplemented by an estimate ``est`` of
:math:`f` at the minimum. Equation 24 gives
:math:`k = 2(\text{est} - f_i)/(p_i' g_i)` on the supposition that the
estimate is right, the minimum lies on the line, and :math:`f` is
quadratic; since that overestimates, equation 25 takes the tentative
step

.. math::

   h = \begin{cases}
     k, & 0 < k < (p_i' p_i)^{-1/2} \\
     (p_i' p_i)^{-1/2}, & \text{otherwise.}
   \end{cases}

Then :math:`\psi'` is examined at :math:`t = 0, h, 2h, 4h, \ldots`,
doubling each time, until the first :math:`b` at which either
:math:`\psi'` is non-negative or :math:`\psi` has not decreased, which
brackets :math:`t_m`; and the bracket is resolved by cubic
interpolation, which uses the function value *and* the slope at each
end and so converges faster than bisection on a smooth function.

Routes kept: ``beta="fletcher-reeves"`` is equation 20 above;
``"polak-ribiere"`` is Polak & Ribiere equation 3.20,

.. math::

   \gamma_i = \frac{\|r_{i+1}\|^2 - r_{i+1}' r_i}{\|r_i\|^2},

written there with :math:`r = -g`, the negative gradient, so in terms
of :math:`g` it is :math:`g_{i+1}'(g_{i+1} - g_i) / (g_i' g_i)` -- the
signs cancel. ``"polak-ribiere-plus"`` adds the ``max(beta, 0)``
safeguard, which is Shewchuk section 14.1 and not in Polak & Ribiere.

Polak & Ribiere prove the two coincide on a quadratic -- "le terme
:math:`r_{i+1}' r_i` est nul lorsque la fonction :math:`f` est
quadratique" -- and the tests use exactly that as a cross-check that
both formulas are right. Their section 4 also confirms the restart rule
from the other side: "Ils proposent, comme optimum, de briser toutes
les :math:`n+1` iterations", and observes that for convex :math:`f`
the modified method does not need the break at all.

``line_search="fletcher-reeves"`` is the three-stage search above;
``"exact-quadratic"`` is available for the quadratic case, where
:math:`t_m = -p'g/(p'Ap)` in closed form.
"""

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = [
    "cgnonl",
    "nonlinear_cg",
    "beta_fletcher_reeves",
    "beta_polak_ribiere",
    "line_search_fr",
    "cubic_interpolate",
    "BETA_RULES",
]

BETA_RULES = ("fletcher-reeves", "polak-ribiere", "polak-ribiere-plus")
_SEARCHES = ("fletcher-reeves", "exact-quadratic")


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


# --------------------------------------------------------------------------
# the beta rules
# --------------------------------------------------------------------------

def beta_fletcher_reeves(g_new, g_old):
    r"""Equation 20: :math:`\beta_i = g_{i+1}' g_{i+1} / (g_i' g_i)`."""
    den = _dot(g_old, g_old)
    if den <= 0.0:
        return 0.0
    return _dot(g_new, g_new) / den


def beta_polak_ribiere(g_new, g_old, plus=False):
    r"""Polak & Ribiere (1969) equation 3.20.

    As printed, with :math:`r = -g`:

    .. math::

       \gamma_i = \frac{\|r_{i+1}\|^2 - r_{i+1}' r_i}{\|r_i\|^2}
                = \frac{g_{i+1}'(g_{i+1} - g_i)}{g_i' g_i}.

    ``plus`` applies the ``max(beta, 0)`` safeguard, which is Shewchuk
    section 14.1 rather than Polak & Ribiere: it restores a convergence
    guarantee, since Polak-Ribiere "can, in rare cases, cycle infinitely
    without converging", though it "often converges much more quickly"
    than Fletcher-Reeves.
    """
    den = _dot(g_old, g_old)
    if den <= 0.0:
        return 0.0
    num = sum(g_new[i] * (g_new[i] - g_old[i]) for i in range(len(g_new)))
    b = num / den
    return max(b, 0.0) if plus else b


def _beta(rule, g_new, g_old):
    if rule == "fletcher-reeves":
        return beta_fletcher_reeves(g_new, g_old)
    if rule == "polak-ribiere":
        return beta_polak_ribiere(g_new, g_old, plus=False)
    if rule == "polak-ribiere-plus":
        return beta_polak_ribiere(g_new, g_old, plus=True)
    raise ValueError("cgnonl: beta must be one of %s" % (BETA_RULES,))


# --------------------------------------------------------------------------
# the line search
# --------------------------------------------------------------------------

def cubic_interpolate(ta, fa, da, tb, fb, db):
    r"""The paper's third stage: cubic fit through value and slope at
    both ends of the bracket.

    Returns the interior stationary point, or the midpoint when the fit
    degenerates.
    """
    h = tb - ta
    if h == 0.0:
        return ta
    z = 3.0 * (fa - fb) / h + da + db
    disc = z * z - da * db
    if disc < 0.0:
        return 0.5 * (ta + tb)
    w = math.sqrt(disc)
    denom = db - da + 2.0 * w
    if denom == 0.0:
        return 0.5 * (ta + tb)
    t = tb - h * (db + w - z) / denom
    if not (min(ta, tb) <= t <= max(ta, tb)):
        return 0.5 * (ta + tb)
    return t


def line_search_fr(f, grad, x, p, f0, g0, est=None, max_double=60,
                   max_cubic=40, tol=1e-12):
    r"""Equations 21-25: the three-stage search along :math:`x + t p`.

    Stage one sets the tentative step :math:`h` from equations 24-25,
    stage two doubles until :math:`\psi'` turns non-negative or
    :math:`\psi` stops decreasing, stage three interpolates.
    Returns ``(t, x_new, f_new, g_new, n_eval)``.
    """
    n = len(x)
    slope0 = _dot(p, g0)
    if slope0 >= 0.0:
        raise ValueError("cgnonl: the search direction is not a descent "
                         "direction (p'g = %g >= 0)" % slope0)
    pnorm = math.sqrt(_dot(p, p))
    if pnorm <= 0.0:
        raise ValueError("cgnonl: the search direction is zero")

    unit = 1.0 / pnorm            # unit length along p in x-space
    if est is None:
        h = unit
    else:
        k = 2.0 * (float(est) - f0) / slope0     # eq. 24
        h = k if 0.0 < k < unit else unit        # eq. 25

    def psi(t):
        xt = [x[i] + t * p[i] for i in range(n)]
        return xt, f(xt), grad(xt)

    evals = 0
    ta, fa, da = 0.0, f0, slope0
    t = h
    tb = fb = db = None
    xb = gb = None
    for _ in range(max_double):
        xt, ft, gt = psi(t)
        evals += 1
        dt = _dot(p, gt)
        if dt >= 0.0 or ft > fa:
            tb, fb, db, xb, gb = t, ft, dt, xt, gt
            break
        ta, fa, da = t, ft, dt
        t *= 2.0
    if tb is None:
        # never bracketed: take the last, best point
        xt, ft, gt = psi(ta)
        return ta, xt, ft, gt, evals + 1

    # stage three: cubic interpolation inside the bracket
    best_t, best_x, best_f, best_g = tb, xb, fb, gb
    if fa < fb:
        xa2, fa2, ga2 = psi(ta)
        evals += 1
        best_t, best_x, best_f, best_g = ta, xa2, fa2, ga2
    for _ in range(max_cubic):
        if abs(tb - ta) <= tol * max(1.0, abs(tb)):
            break
        tc = cubic_interpolate(ta, fa, da, tb, fb, db)
        xc, fc, gc = psi(tc)
        evals += 1
        dc = _dot(p, gc)
        if fc < best_f:
            best_t, best_x, best_f, best_g = tc, xc, fc, gc
        if abs(dc) <= tol * max(1.0, abs(slope0)):
            return tc, xc, fc, gc, evals
        if dc < 0.0:
            ta, fa, da = tc, fc, dc
        else:
            tb, fb, db = tc, fc, dc
    return best_t, best_x, best_f, best_g, evals


def _exact_quadratic_step(x, p, g, hess_vec):
    r""":math:`t_m = -p'g / (p'Ap)`, exact for a quadratic."""
    ap = hess_vec(p)
    den = _dot(p, ap)
    if den <= 0.0:
        raise ValueError("cgnonl: p'Ap = %g is not positive; the exact "
                         "quadratic step needs a positive definite A"
                         % den)
    return -_dot(p, g) / den


# --------------------------------------------------------------------------
# the method itself
# --------------------------------------------------------------------------

def nonlinear_cg(f, grad, x0, beta="fletcher-reeves", restart=None,
                 max_iter=None, tol=1e-10, est=None,
                 line_search="fletcher-reeves", hess_vec=None,
                 keep_path=False):
    r"""Fletcher & Reeves' equation 20, with their restart rule.

    Parameters
    ----------
    f, grad : callable
        :math:`f(x)` and :math:`g(x)`, the value and gradient.
    x0 : sequence
        The starting point. The paper notes this matters for general
        functions -- the method finds the bottom of whatever valley it
        starts in -- and is immaterial for quadratics.
    beta : {"fletcher-reeves", "polak-ribiere", "polak-ribiere-plus"}
        Equation 20, or Shewchuk section 14.1 with and without the
        ``max(beta, 0)`` safeguard.
    restart : int, optional
        Revert to steepest descent every this many iterations.
        Defaults to :math:`n+1`, the paper's choice; quadratic
        convergence needs restarts no more frequent than every
        :math:`n`. ``0`` disables restarts.
    est : float, optional
        An estimate of :math:`f` at the minimum, used by equations
        24-25 to set the tentative step.
    line_search : {"fletcher-reeves", "exact-quadratic"}
        The three-stage search, or the closed-form step for a
        quadratic, which needs ``hess_vec``.
    hess_vec : callable, optional
        :math:`p \mapsto Ap`, for ``line_search="exact-quadratic"``.

    Returns
    -------
    RichResult
        ``x``, ``fun``, ``grad``, ``n_iter``, ``n_restart``,
        ``n_feval``, ``converged``, and the ``beta`` history.
    """
    if beta not in BETA_RULES:
        raise ValueError("cgnonl: beta must be one of %s" % (BETA_RULES,))
    if line_search not in _SEARCHES:
        raise ValueError("cgnonl: line_search must be one of %s"
                         % (_SEARCHES,))
    if line_search == "exact-quadratic" and hess_vec is None:
        raise ValueError("cgnonl: line_search='exact-quadratic' needs "
                         "hess_vec, the map p -> Ap")
    x = [float(v) for v in x0]
    n = len(x)
    if n == 0:
        raise ValueError("cgnonl: x0 is empty")
    if restart is None:
        restart = n + 1
    restart = int(restart)
    if restart < 0:
        raise ValueError("cgnonl: restart must not be negative")
    if max_iter is None:
        max_iter = 200 * n
    if int(max_iter) < 1:
        raise ValueError("cgnonl: max_iter must be at least 1")

    g = [float(v) for v in grad(x)]
    fx = float(f(x))
    p = [-v for v in g]
    evals = 1
    betas = []
    path = [list(x)] if keep_path else []
    restarts = 0
    it = 0
    converged = _dot(g, g) <= tol * tol

    while not converged and it < int(max_iter):
        it += 1
        if line_search == "exact-quadratic":
            t = _exact_quadratic_step(x, p, g, hess_vec)
            x_new = [x[i] + t * p[i] for i in range(n)]
            f_new = float(f(x_new))
            g_new = [float(v) for v in grad(x_new)]
            evals += 1
        else:
            t, x_new, f_new, g_new, ev = line_search_fr(
                f, grad, x, p, fx, g, est=est)
            evals += ev

        g_old = g
        x, fx, g = x_new, f_new, [float(v) for v in g_new]
        if keep_path:
            path.append(list(x))
        if _dot(g, g) <= tol * tol:
            converged = True
            break

        if restart and it % restart == 0:
            # "revert ... to the steepest descent direction -g in place
            # of the customary p ... discarding all previous experience"
            p = [-v for v in g]
            restarts += 1
            betas.append(0.0)
        else:
            b = _beta(beta, g, g_old)
            betas.append(b)
            p = [-g[i] + b * p[i] for i in range(n)]
            if _dot(p, g) >= 0.0:
                # not a descent direction; fall back to steepest descent
                p = [-v for v in g]
                restarts += 1

    return RichResult(payload={
        "x": x,
        "fun": fx,
        "grad": g,
        "gnorm": math.sqrt(_dot(g, g)),
        "n_iter": it,
        "n_restart": restarts,
        "n_feval": evals,
        "converged": bool(converged),
        "betas": betas,
        "path": path,
        "beta_rule": beta,
        "line_search": line_search,
        "restart_every": restart,
        "method": ("Fletcher & Reeves (1964) eq. 20, nonlinear conjugate "
                   "gradients"),
        "note": ("storage is three vectors -- x, g and p -- which is the "
                 "paper's stated advantage over Davidon-Fletcher-Powell; "
                 "restarts to steepest descent every n+1 iterations, "
                 "which preserves quadratic convergence because they are "
                 "no more frequent than every n"),
    })


cgnonl = nonlinear_cg


def cheatsheet():
    return ("cgnonl: nonlinear conjugate gradients, Fletcher & Reeves "
            "(1964) eq. 20. p_0 = -g_0; search the line through x_i "
            "along p_i; beta_i = g'_{i+1} g_{i+1} / (g'_i g_i); "
            "p_{i+1} = -g_{i+1} + beta_i p_i. Guaranteed to minimise "
            "any quadratic in n variables in at most n iterations, and "
            "it stores only three vectors. Restarts to steepest descent "
            "every n+1 iterations -- their fix for Rosenbrock's valley, "
            "where successive directions went nearly parallel. "
            "beta='polak-ribiere' is Polak & Ribiere (1969) eq. 3.20, "
            "gamma = (|r_{i+1}|^2 - r'_{i+1} r_i) / |r_i|^2; the "
            "'-plus' max(beta, 0) safeguard is Shewchuk sec. 14.1. On a "
            "quadratic the two coincide, which that paper proves.")
