r"""Limited-memory BFGS.

Liu, D. C., & Nocedal, J. (1989) "On the limited memory BFGS method for
large scale optimization", *Mathematical Programming* **45**, 503-528.

The search direction is formed by the two-loop recursion (their Sec. 2),
which applies the inverse Hessian approximation implicitly from the last
``m`` correction pairs :math:`(s_k, y_k)`, never forming a matrix:

    q = g
    for i = k-1 .. k-m:   alpha_i = rho_i s_i' q ;  q -= alpha_i y_i
    r = H0 q                       H0 = (s'y / y'y) I    (Sec. 2, eq. 7)
    for i = k-m .. k-1:   beta = rho_i y_i' r ;  r += (alpha_i - beta) s_i

with :math:`\rho_i = 1/(y_i's_i)`. Pairs with :math:`y's \le 0` are
skipped: accepting them would destroy positive-definiteness and the
direction would stop being a descent direction.

The line search enforces the **Wolfe** conditions, not merely Armijo:

.. math:: f(x + td) \le f(x) + c_1 t g'd, \qquad
          \nabla f(x + td)'d \ge c_2\, g'd .

Armijo alone is *not* enough, and the failure is not subtle: without the
curvature condition :math:`y's` goes negative within a handful of
iterations, every correction pair is then rejected by the guard above,
the memory freezes and the method degenerates into a fixed-direction
crawl. The curvature condition is precisely what makes :math:`y's > 0`,
so the two conditions are load-bearing together.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["lbfgs_minimize", "lbfgsm"]


def _dot(a, b):
    return sum(float(u) * float(v) for u, v in zip(a, b))


_TOL_TYPES = ("absolute", "relative", "initial")


def _precond(precond, S, Y, n):
    """Diagonal preconditioner for the two-loop recursion.

    ``None`` leaves H0 = gamma I; ``"auto"`` scales each coordinate by
    |y_i| / |s_i| from the most recent pair, normalised to unit geometric
    mean; a numeric vector is used as given (one positive entry per
    parameter). Same rule as the R arm's ``.lbfgsm_precond``.
    """
    if precond is None:
        return None
    if isinstance(precond, str):
        if precond != "auto":
            raise ValueError("`precond` must be None, \"auto\", or a numeric vector")
        if not S:
            return None
        s, y = S[-1], Y[-1]
        eps = 2.220446049250313e-16
        d = [abs(yi) / max(abs(si), eps) for si, yi in zip(s, y)]
        d = [v if (math.isfinite(v) and v > 0) else 1.0 for v in d]
        lg = sum(math.log(v) for v in d) / len(d)
        if not math.isfinite(lg):
            return None
        d = [v / math.exp(lg) for v in d]
        return [min(max(v, 1e-8), 1e8) for v in d]
    d = [float(v) for v in np.atleast_1d(np.asarray(precond, dtype=float))]
    if len(d) != n:
        raise ValueError(f"`precond` must have one entry per parameter ({n}), got {len(d)}")
    if any((not math.isfinite(v)) or v <= 0 for v in d):
        raise ValueError("`precond` must be positive and finite")
    return d


def lbfgs_minimize(fun, x0, grad, m=10, max_iter=200, tol=1e-8,
                   tol_type="absolute", precond=None,
                   c1=1e-4, c2=0.9, max_ls=60):
    r"""Minimise ``fun`` from ``x0`` using L-BFGS with memory ``m``.

    ``tol_type`` chooses what ``tol`` bounds: the gradient norm
    (``"absolute"``), the gradient norm divided by 1 + |f| (``"relative"``),
    or the gradient norm divided by its value at ``x0`` (``"initial"``), so
    the stopping rule is scale-aware. ``precond`` is None, ``"auto"`` or a
    positive vector (see ``_precond``). The result carries ``status``, a
    sentence saying why the iteration stopped.
    """
    if tol_type not in _TOL_TYPES:
        raise ValueError(f"lbfgs_minimize: tol_type must be one of {_TOL_TYPES!r}, got {tol_type!r}")
    x = [float(v) for v in np.atleast_1d(np.asarray(x0, dtype=float))]
    n = len(x)
    m = int(m)
    if m < 1:
        raise ValueError(f"lbfgs_minimize: m must be at least 1, got {m!r}")
    if n == 0:
        raise ValueError("lbfgs_minimize: x0 must be non-empty")

    f = float(fun(x))
    g = [float(v) for v in grad(x)]
    S, Y, RHO = [], [], []
    n_f = 1
    it = 0
    converged = False
    status = "iteration limit reached"
    gnorm0 = math.sqrt(_dot(g, g))
    history = [f]

    for it in range(1, int(max_iter) + 1):
        gnorm = math.sqrt(_dot(g, g))
        if it == 1:
            gnorm0 = gnorm
        if tol_type == "absolute":
            crit = gnorm
        elif tol_type == "relative":
            crit = gnorm / (1.0 + abs(f))
        else:
            crit = gnorm / gnorm0 if gnorm0 > 0 else 0.0
        if crit <= tol:
            converged = True
            status = {"absolute": "gradient norm within tolerance",
                      "relative": "gradient norm within tolerance, relative to f",
                      "initial": "gradient norm within tolerance, relative to its start"}[tol_type]
            break

        # --- two-loop recursion, Liu & Nocedal Sec. 2
        q = list(g)
        alphas = []
        for i in range(len(S) - 1, -1, -1):
            a = RHO[i] * _dot(S[i], q)
            alphas.append(a)
            for j in range(n):
                q[j] -= a * Y[i][j]
        # H0 = (s'y / y'y) I -- the scaling that makes L-BFGS work at
        # all; with H0 = I the first step is wildly mis-scaled.
        gamma = _dot(S[-1], Y[-1]) / _dot(Y[-1], Y[-1]) if S else 1.0
        dvec = _precond(precond, S, Y, n)
        r = [gamma * v for v in q] if dvec is None else [gamma * (v / dv) for v, dv in zip(q, dvec)]
        alphas.reverse()
        for i in range(len(S)):
            b = RHO[i] * _dot(Y[i], r)
            coef = alphas[i] - b
            for j in range(n):
                r[j] += coef * S[i][j]
        d = [-v for v in r]

        slope = _dot(g, d)
        if slope >= 0.0:
            # Numerically lost descent; reset the memory and go downhill.
            S, Y, RHO = [], [], []
            d = [-v for v in g]
            slope = -_dot(g, g)

        # --- Wolfe line search by bracketing. Widen while the
        # curvature condition fails, bisect while Armijo fails.
        lo, hi = 0.0, float("inf")
        t = 1.0
        ok = False
        xt, ft, gt = None, None, None
        # Armijo with a rounding allowance: below the evaluation noise of f
        # the plain test rejects every step at the optimum.
        ftol = 4 * 2.220446049250313e-16 * max(1.0, abs(f))
        for _ in range(int(max_ls)):
            xt = [x[j] + t * d[j] for j in range(n)]
            ft = float(fun(xt))
            n_f += 1
            if ft > f + c1 * t * slope + ftol:
                hi = t
                t = 0.5 * (lo + hi)
                continue
            gt = [float(v) for v in grad(xt)]
            if _dot(gt, d) < c2 * slope:
                lo = t
                t = 2.0 * lo if hi == float("inf") else 0.5 * (lo + hi)
                continue
            ok = True
            break
        if not ok:
            if S:
                # the curvature pairs may be stale: forget them and retry
                # along the steepest descent direction
                S, Y, RHO = [], [], []
                d = [-v for v in g]
                slope = -_dot(g, g)
                lo, hi, t = 0.0, float("inf"), 1.0
                for _ in range(int(max_ls)):
                    xt = [x[j] + t * d[j] for j in range(n)]
                    ft = float(fun(xt))
                    n_f += 1
                    if ft > f + c1 * t * slope:
                        hi = t
                        t = 0.5 * (lo + hi)
                        continue
                    gt = [float(v) for v in grad(xt)]
                    if _dot(gt, d) < c2 * slope:
                        lo = t
                        t = 2.0 * lo if hi == float("inf") else 0.5 * (lo + hi)
                        continue
                    ok = True
                    break
            if not ok:
                if math.sqrt(_dot(g, g)) <= math.sqrt(2.220446049250313e-16) * max(1.0, abs(f)):
                    status = "stalled at the optimum; no Wolfe step remains"
                else:
                    status = "line search could not satisfy the Wolfe conditions"
                break
        if gt is None:
            gt = [float(v) for v in grad(xt)]
        s = [xt[j] - x[j] for j in range(n)]
        y = [gt[j] - g[j] for j in range(n)]
        ys = _dot(y, s)
        if ys > 1e-16:
            S.append(s)
            Y.append(y)
            RHO.append(1.0 / ys)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
                RHO.pop(0)
        x, f, g = xt, ft, gt
        history.append(f)

    return RichResult(payload={
        "estimate": x,
        "x": x,
        "fun": float(f),
        "grad": g,
        "grad_norm": float(math.sqrt(_dot(g, g))),
        "grad_norm_relative": float(math.sqrt(_dot(g, g)) / (1.0 + abs(f))),
        "grad_norm_ratio": (float(math.sqrt(_dot(g, g)) / gnorm0) if gnorm0 > 0 else None),
        "tol": float(tol),
        "tol_type": tol_type,
        "iterations": int(it),
        "n_fun": int(n_f),
        "memory": int(m),
        "converged": bool(converged),
        "status": status,
        "history": history,
        "method": "L-BFGS two-loop recursion with a Wolfe line search "
                  "(Liu & Nocedal 1989, Sec. 2)",
    })


def cheatsheet():
    return ("lbfgsm: L-BFGS two-loop recursion, H0 = (s'y/y'y) I, curvature "
            "pairs with y's <= 0 skipped, Armijo backtracking.")


lbfgsm = lbfgs_minimize
