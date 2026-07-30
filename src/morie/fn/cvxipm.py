# morie.fn -- function file (rootcoder007/morie)
"""Barrier method (interior point) -- Boyd & Vandenberghe Sec. 11.3."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_interior_point"]


def _num_grad_hess(fn, x, h=1e-05):
    """Central-difference gradient and Hessian of a scalar function."""
    n = x.size
    step = h * np.maximum(1.0, np.abs(x))
    g = np.empty(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = step[i]
        g[i] = (fn(x + e) - fn(x - e)) / (2.0 * step[i])
    H = np.empty((n, n))
    f0 = fn(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step[i]
        H[i, i] = (fn(x + ei) - 2.0 * f0 + fn(x - ei)) / step[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step[j]
            H[i, j] = H[j, i] = (
                fn(x + ei + ej) - fn(x + ei - ej)
                - fn(x - ei + ej) + fn(x - ei - ej)
            ) / (4.0 * step[i] * step[j])
    return g, H


def _centering(f0, fs, x, t, tol=1e-12, max_iter=200):
    r"""Newton centering: minimise ``t f0(x) - sum log(-f_i(x))``."""

    def phi(z):
        vals = np.array([float(fi(z)) for fi in fs]) if fs else np.zeros(0)
        if fs and np.any(vals >= 0):
            return np.inf
        return float(t * float(f0(z))
                     - (np.sum(np.log(-vals)) if fs else 0.0))

    for it in range(int(max_iter)):
        # A fixed difference step stalls the moment the centered point
        # gets closer to the boundary than the step itself: the probes
        # straddle it, the barrier reads +inf, and the method stops
        # dead well short of the optimum. Tie the step to the slack.
        if fs:
            slack = float(np.min([-float(fi(x)) for fi in fs]))
            scale = max(1.0, float(np.max(np.abs(x))))
            h = min(1e-05, max(1e-09, 0.1 * slack / scale))
        else:
            h = 1e-05
        g, H = _num_grad_hess(phi, x, h)
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            break
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(H, g, rcond=None)[0]
        lam2 = float(-g @ step)
        if not np.isfinite(lam2) or lam2 / 2.0 <= tol:
            return x, it, True
        cur = phi(x)
        s = 1.0
        while s > 1e-16 and not (phi(x + s * step) <= cur - 0.25 * s * lam2):
            s *= 0.5
        if s <= 1e-16:
            return x, it, True
        x = x + s * step
    return x, max_iter, False


def boyd_interior_point(f0, f=(), x0=None, t=1.0, mu=10.0, tol=1e-06,
                        max_outer=100):
    r"""Barrier method: minimise :math:`f_0` s.t. :math:`f_i(x) \le 0`.

    Replaces each inequality with the logarithmic barrier
    :math:`-\tfrac1t\log(-f_i(x))`, which is :math:`+\infty` outside the
    feasible set and vanishes as :math:`t` grows, and solves the
    resulting smooth problem by Newton's method for an increasing
    sequence of :math:`t`.

    The reason to prefer this to a penalty method is the exact
    suboptimality bound it comes with. The centered point
    :math:`x^{\star}(t)` satisfies

    .. math::

        f_0(x^{\star}(t)) - p^{\star} \le m/t,

    with :math:`m` the number of constraints -- so the stopping rule is
    a CERTIFICATE, not a heuristic, and the accuracy is known before the
    optimum is. Every iterate is strictly feasible, which is the other
    half of the bargain: the method can be stopped at any point and its
    current answer is usable.

    Parameters
    ----------
    f0 : callable
        Objective ``f0(x) -> float``.
    f : sequence of callable
        Inequality constraints, each ``f_i(x) <= 0``. May be empty, in
        which case this is plain Newton minimisation.
    x0 : array-like
        STRICTLY feasible start (all ``f_i(x0) < 0``). Required: finding
        one is a separate phase-I problem.
    t : float
        Initial barrier parameter.
    mu : float
        Multiplier applied to ``t`` each outer iteration.
    tol : float
        Stop when ``m / t`` falls below this. The default is 1e-6 rather
        than machine-ish because derivatives here are finite
        differences: as the centered point approaches the boundary the
        slack shrinks, the probe points straddle it, and no smaller step
        buys accuracy. Pass a tighter value only with an objective whose
        curvature is mild near the active set.
    max_outer : int
        Cap on outer iterations.

    Returns
    -------
    RichResult
        ``x``, ``objective``, ``gap_bound``, ``path`` (one row per outer
        iteration), ``t_values``, ``constraints`` (final ``f_i(x)``),
        ``strictly_feasible``, ``newton_steps``, ``outer``,
        ``converged``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.
    Nesterov, Y., & Nemirovskii, A. (1994). *Interior-Point Polynomial
        Algorithms in Convex Programming*. SIAM.

    Examples
    --------
    Minimise ``x^2/2`` subject to ``x >= 1``. The optimum is at the
    boundary, ``x = 1``, with value ``1/2``.

    >>> import numpy as np
    >>> obj = lambda x: 0.5 * x[0] ** 2
    >>> con = [lambda x: 1.0 - x[0]]
    >>> r = boyd_interior_point(obj, con, [2.0])
    >>> round(float(r["x"][0]), 5), round(float(r["objective"]), 5)
    (1.0, 0.5)

    Stop after ONE outer iteration and the answer is the centered point
    for ``t = 1``, which here has a closed form: the stationarity
    condition ``t*x = 1/(x-1)`` becomes ``x^2 - x - 1 = 0``, so
    ``x*(1)`` is the golden ratio. The barrier keeps it strictly inside.

    >>> one = boyd_interior_point(obj, con, [2.0], max_outer=1)
    >>> round(float(one["x"][0]), 6)
    1.618034
    >>> round(float((1 + 5 ** 0.5) / 2), 6)
    1.618034

    That point is m/t = 1 above the optimum at worst, and the bound is
    honest -- the true suboptimality here is 0.809.

    >>> round(float(one["gap_bound"]), 6)
    1.0
    >>> bool(one["objective"] - 0.5 <= one["gap_bound"])
    True

    Every iterate along the path is STRICTLY feasible -- the barrier is
    infinite on the boundary, so the method can never step onto or over
    it. That is what makes an early stop safe.

    >>> bool(np.all(r["path"] > 1.0))
    True

    An inactive constraint costs nothing: the barrier's pull vanishes as
    t grows, so the answer is the unconstrained minimiser.

    >>> free = boyd_interior_point(obj, [lambda x: x[0] - 5.0], [1.0])
    >>> bool(abs(free["x"][0]) < 1e-06)
    True

    A start that is merely feasible, not STRICTLY so, is refused: the
    barrier is undefined there, and phase I is a separate problem rather
    than something to fake with a nudge.

    >>> boyd_interior_point(obj, con, [1.0])
    Traceback (most recent call last):
        ...
    ValueError: x0 is not strictly feasible: constraint 0 has f(x0) = 0
    """
    if not callable(f0):
        raise TypeError("f0 must be callable")
    fs = list(f)
    for k, fi in enumerate(fs):
        if not callable(fi):
            raise TypeError(f"f[{k}] must be callable")
    if x0 is None:
        raise ValueError("x0 is required: the barrier method needs a "
                         "strictly feasible start (phase I)")
    x = np.atleast_1d(np.asarray(x0, dtype=float)).ravel().copy()
    for k, fi in enumerate(fs):
        v = float(fi(x))
        if not (v < 0):
            raise ValueError(
                f"x0 is not strictly feasible: constraint {k} has "
                f"f(x0) = {v:g}")
    t = float(t)
    if t <= 0:
        raise ValueError(f"t must be positive, got {t}")
    if mu <= 1:
        raise ValueError(f"mu must exceed 1, got {mu}")
    m = len(fs)
    path, ts, steps = [], [], []
    converged = m == 0
    for _ in range(int(max_outer)):
        x, it, _ok = _centering(f0, fs, x, t)
        path.append(x.copy())
        ts.append(t)
        steps.append(int(it))
        if m == 0 or m / t < tol:
            converged = True
            break
        t *= mu
    cons = np.array([float(fi(x)) for fi in fs]) if m else np.zeros(0)
    # The bound belongs to the t the final point was CENTERED at, not to
    # the t the loop had already advanced to on its way out; reporting
    # the latter would understate the error by a factor of mu.
    gap = float(m / ts[-1]) if m else 0.0
    return RichResult(
        title="Barrier method",
        summary_lines=[("n", int(x.size)), ("constraints", int(m)),
                       ("objective", float(f0(x))),
                       ("gap bound", gap),
                       ("outer", len(path))],
        payload={
            "x": x, "objective": float(f0(x)),
            "gap_bound": gap,
            "path": np.array(path), "t_values": np.array(ts),
            "constraints": cons,
            "strictly_feasible": bool(np.all(cons < 0)) if m else True,
            "newton_steps": np.array(steps), "outer": len(path),
            "t": t, "converged": bool(converged),
            "method": "boyd_interior_point",
        },
    )


def cheatsheet():
    return "cvxipm: m/t is a CERTIFICATE, not a heuristic -- and every iterate is strictly feasible, so early stops are safe"
