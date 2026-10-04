# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""
ADMM (alternating direction method of multipliers) for convex optimization.

Splits problem into subproblems; solves via alternating proximal updates.
"""

from . import _array_core as np

__all__ = ["admm"]


def admm(f, g, A, b, rho=1.0, max_iter=1000, tol=1e-4, full_output=False, prox_g=None):
    """
    ADMM for minimize f(x) + g(z) subject to Ax + z = b.

    Scaled form of Boyd et al. (2011), section 3.1.1, with B = I and c = b:

    - x-update: x = argmin f(x) + (rho/2) ||Ax + z - b + u||^2 (numerically)
    - z-update: z = prox_{g/rho}(b - Ax - u)
    - u-update: u = u + Ax + z - b

    It stops when the primal residual ||Ax + z - b|| and the dual residual
    rho ||A^T (z - z_old)|| are both below ``tol`` (section 3.3.1).

    Parameters
    ----------
    f : callable
        Function f(x) to minimize (convex).
    g : callable
        Function g(z) to minimize (convex).
    A : ndarray
        Constraint matrix A (m, n).
    b : ndarray
        Constraint right-hand side (m,).
    rho : float, optional
        Penalty parameter (default 1.0).
    max_iter : int, optional
        Maximum iterations (default 1000).
    tol : float, optional
        Tolerance on the primal and dual residuals (default 1e-4).
    full_output : bool, optional
        If True, return (x, info_dict).
    prox_g : callable, optional
        ``prox_g(v, t)`` = argmin_z g(z) + ||z - v||^2 / (2 t). Without it the
        proximal step is solved numerically from the soft-threshold of ``v``
        (exact for g = ||z||_1).

    Returns
    -------
    x : ndarray
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'iterations', 'converged', 'z', 'primal_residual', 'dual_residual'.

    References
    ----------
    Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
    Distributed optimization and statistical learning via ADMM. Foundations
    and Trends in Machine Learning, 3(1), 1-122.

    Examples
    --------
    minimize ||x||^2 + ||z||_1 subject to x + z = b: for each b_i > 1/2 the
    solution is x_i = 1/2 (the slope of x^2 meets that of |b - x|).

    >>> from morie.fn import _array_core as np
    >>> from morie.fn import admm
    >>> f = lambda x: np.sum(x**2)
    >>> g = lambda z: np.sum(np.abs(z))
    >>> A = np.eye(3)
    >>> b = np.array([1, 2, 3])
    >>> x, info = admm(f, g, A, b, full_output=True)
    >>> info['converged']
    True
    >>> [round(float(v), 3) for v in x]
    [0.5, 0.5, 0.5]
    """
    from ._sci_core import minimize

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(m)
    u = np.zeros(m)
    t = 1.0 / rho

    def prox(v):
        if prox_g is not None:
            return np.asarray(prox_g(v, t), dtype=float)
        start = np.sign(v) * np.maximum(np.abs(v) - t, 0)  # the exact answer when g is the L1 norm
        res = minimize(lambda z_: g(z_) + np.sum((z_ - v) ** 2) / (2 * t), start, method="Nelder-Mead")
        cand = np.asarray(res.x, dtype=float)
        # keep the start when the search did not improve on it
        obj = lambda z_: g(z_) + np.sum((z_ - v) ** 2) / (2 * t)  # noqa: E731 - local objective
        return cand if obj(cand) < obj(start) else start

    r_norm = s_norm = float("inf")
    for it in range(max_iter):
        res = minimize(lambda x_, z=z, u=u: f(x_) + 0.5 * rho * np.sum((A @ x_ + z - b + u) ** 2), x, method="BFGS")
        x = np.asarray(res.x, dtype=float)
        z_old = z
        z = prox(b - A @ x - u)
        r = A @ x + z - b
        u = u + r
        r_norm = float(np.linalg.norm(r))
        s_norm = float(rho * np.linalg.norm(A.T @ (z - z_old)))
        if r_norm < tol and s_norm < tol:
            info = {"iterations": it + 1, "converged": True, "z": z, "primal_residual": r_norm, "dual_residual": s_norm}
            return (x, info) if full_output else x

    info = {"iterations": max_iter, "converged": False, "z": z, "primal_residual": r_norm, "dual_residual": s_norm}
    return (x, info) if full_output else x


def cheatsheet() -> str:
    return "admm: admm(f, g, A, b, rho, max_iter, tol, full_output) -> ADMM for constrained optimization."
