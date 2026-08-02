# morie.fn -- function file (rootcoder007/morie)
"""Dual optimization problem -- Boyd & Vandenberghe Sec. 5.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_dual_problem"]


def boyd_dual_problem(g, n_lambda=0, n_nu=0, lambda0=None, nu0=None,
                      primal_value=None, check_concavity=True, seed=0):
    r"""Maximise the dual function: :math:`\max g(\lambda,\nu)` s.t.
    :math:`\lambda \succeq 0`.

    Where :func:`boyd_dual_function` EVALUATES the dual at a point, this
    solves the dual problem itself and returns the best lower bound the
    dual can certify. The two facts that make that worth doing:

    * :math:`g` is concave in :math:`(\lambda,\nu)` for ANY primal,
      convex or not, because it is a pointwise infimum of functions
      affine in the multipliers. So the dual problem is always convex
      even when the primal is hopeless.
    * :math:`g(\lambda,\nu) \le p^{\star}` for every feasible
      :math:`\lambda \succeq 0`. Its maximum :math:`d^{\star}` is the
      best such bound; :math:`p^{\star} - d^{\star}` is the optimal
      duality gap, zero under Slater's condition and generally positive
      otherwise.

    The sign asymmetry is structural: :math:`\lambda` is constrained
    nonnegative because an inequality can only be violated in one
    direction, while :math:`\nu` is free because an equality is
    two-sided.

    Parameters
    ----------
    g : callable
        ``g(lambda_, nu) -> float``, both arguments 1-d arrays (possibly
        empty). May return ``-inf`` where the dual is vacuous.
    n_lambda, n_nu : int
        Number of inequality and equality multipliers.
    lambda0, nu0 : array-like, optional
        Starting point. Defaults to zeros, which is always dual feasible.
    primal_value : float, optional
        Known ``p*``. If given, the optimal duality gap is reported.
    check_concavity : bool
        Sample midpoints and verify ``g(mid) >= mean`` -- catches a ``g``
        that is not a dual function at all.
    seed : int
        RNG seed for the concavity probe.

    Returns
    -------
    RichResult
        ``lambda_``, ``nu``, ``dual_value``, ``active``,
        ``bound_improves`` (whether the optimum beats the trivial
        ``lambda = 0`` bound), ``duality_gap``, ``strong_duality``,
        ``concave``, ``converged``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    The dual of ``min |x|^2/2 - 1'x`` s.t. ``x1 + x2 <= 1`` works out to
    ``g(lam) = -lam^2 + lam - 1``. Its maximum is the primal optimum.

    >>> g = lambda lam, nu: -lam[0] ** 2 + lam[0] - 1.0
    >>> r = boyd_dual_problem(g, n_lambda=1, primal_value=-0.75)
    >>> round(float(r["lambda_"][0]), 6), round(float(r["dual_value"]), 6)
    (0.5, -0.75)
    >>> round(float(r["duality_gap"]), 6), bool(r["strong_duality"])
    (0.0, True)

    The bound is worth having only because it beats the free one: any
    ``lambda >= 0`` certifies something, and ``lambda = 0`` certifies the
    unconstrained minimum. Here the optimum improves on it.

    >>> bool(r["bound_improves"])
    True

    When the unconstrained maximum of g sits at a NEGATIVE multiplier,
    the constraint binds and the optimum is at zero -- the primal
    constraint was slack, so the dual paid nothing for it.

    >>> slack = boyd_dual_problem(lambda lam, nu: -lam[0] ** 2 - lam[0] - 1.0,
    ...                           n_lambda=1)
    >>> round(float(slack["lambda_"][0]), 9), round(float(slack["dual_value"]), 6)
    (0.0, -1.0)
    >>> bool(slack["bound_improves"])
    False

    Concavity holds whether or not the primal is convex -- that is the
    property being probed, and a g that fails it is not a dual function.

    >>> bool(r["concave"])
    True
    >>> bool(boyd_dual_problem(lambda lam, nu: lam[0] ** 2, n_lambda=1,
    ...                        check_concavity=True)["concave"])
    False

    Equality multipliers are unconstrained in sign, so a nu-only dual is
    free to walk negative.

    >>> e = boyd_dual_problem(lambda lam, nu: -(nu[0] + 2.0) ** 2, n_nu=1)
    >>> round(float(e["nu"][0]), 6)
    -2.0
    """
    from scipy.optimize import minimize

    n_lambda = int(n_lambda)
    n_nu = int(n_nu)
    if n_lambda < 0 or n_nu < 0:
        raise ValueError("n_lambda and n_nu must be nonnegative")
    if n_lambda + n_nu == 0:
        raise ValueError("no multipliers: the dual problem is empty")
    if not callable(g):
        raise TypeError("g must be callable as g(lambda_, nu)")
    lam0 = (np.zeros(n_lambda) if lambda0 is None
            else np.atleast_1d(np.asarray(lambda0, dtype=float)).ravel())
    nu_0 = (np.zeros(n_nu) if nu0 is None
            else np.atleast_1d(np.asarray(nu0, dtype=float)).ravel())
    if lam0.size != n_lambda or nu_0.size != n_nu:
        raise ValueError("lambda0/nu0 length does not match n_lambda/n_nu")
    if np.any(lam0 < 0):
        raise ValueError("lambda0 must be nonnegative to be dual feasible")

    def split(z):
        return z[:n_lambda], z[n_lambda:]

    def neg(z):
        lam, nu = split(z)
        val = float(g(np.maximum(lam, 0.0), nu))
        # A vacuous bound is legitimate output from g, but -inf derails
        # a quasi-Newton line search; a large finite penalty keeps the
        # iterate moving back toward the useful region.
        return 1e300 if not np.isfinite(val) else -val

    z0 = np.r_[lam0, nu_0]
    bounds = [(0.0, None)] * n_lambda + [(None, None)] * n_nu
    res = minimize(neg, z0, method="L-BFGS-B", bounds=bounds,
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000})
    lam, nu = split(np.asarray(res.x, dtype=float))
    lam = np.maximum(lam, 0.0)
    dual_val = float(g(lam, nu))
    trivial = float(g(np.zeros(n_lambda), np.zeros(n_nu)))

    concave = None
    if check_concavity:
        rng = np.random.default_rng(seed)
        ok = True
        scale = max(1.0, float(np.max(np.abs(np.r_[lam, nu]))) * 2.0)
        for _ in range(40):
            a = np.r_[np.abs(rng.normal(scale=scale, size=n_lambda)),
                      rng.normal(scale=scale, size=n_nu)]
            b = np.r_[np.abs(rng.normal(scale=scale, size=n_lambda)),
                      rng.normal(scale=scale, size=n_nu)]
            va, vb = float(g(*split(a))), float(g(*split(b)))
            vm = float(g(*split(0.5 * (a + b))))
            if not (np.isfinite(va) and np.isfinite(vb)):
                continue
            if vm < 0.5 * (va + vb) - 1e-08 * max(1.0, abs(va) + abs(vb)):
                ok = False
                break
        concave = ok

    gap = None if primal_value is None else float(primal_value) - dual_val
    return RichResult(
        title="Dual problem",
        summary_lines=[("n_lambda", n_lambda), ("n_nu", n_nu),
                       ("dual optimum", dual_val),
                       ("active", int(np.sum(lam > 1e-08))),
                       ("gap", gap if gap is not None else float("nan"))],
        payload={
            "lambda_": lam, "nu": nu, "dual_value": dual_val,
            "active": lam > 1e-08,
            "trivial_bound": trivial,
            "bound_improves": bool(dual_val > trivial + 1e-09),
            "duality_gap": gap,
            "strong_duality": (None if gap is None
                               else bool(abs(gap) < 1e-06
                                         * max(1.0, abs(dual_val)))),
            "concave": concave,
            "converged": bool(res.success),
            "method": "boyd_dual_problem",
        },
    )


def cheatsheet():
    return "cvxdgp: dual is CONCAVE even for a nonconvex primal; lambda >= 0, nu free -- inequalities are one-sided"
