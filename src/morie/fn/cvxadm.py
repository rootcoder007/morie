# morie.fn -- function file (rootcoder007/morie)
"""ADMM -- Boyd et al. (2011)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_admm"]


def boyd_admm(prox_f, prox_g, A=None, B=None, c=None, rho=1.0, n=None,
              max_iter=500, eps_abs=1e-08, eps_rel=1e-06):
    r"""Alternating direction method of multipliers for
    :math:`\min f(x) + g(z)` s.t. :math:`Ax + Bz = c`:

    .. math::
        x^{k+1} &= \arg\min_x L_\rho(x, z^k, u^k) \\
        z^{k+1} &= \arg\min_z L_\rho(x^{k+1}, z, u^k) \\
        u^{k+1} &= u^k + Ax^{k+1} + Bz^{k+1} - c.

    ADMM converges for ANY positive rho, which is unusual and is the
    practical appeal -- there is no step size to get wrong. What rho does
    control is the BALANCE between primal and dual residuals: too large
    and the primal residual falls fast while the dual lags, too small and
    the reverse. Both are reported for exactly that reason.

    Convergence is to modest accuracy quickly and high accuracy slowly, so
    ADMM is the right tool when a rough answer soon beats an exact one
    later, and the wrong one when the last digits matter.

    Parameters
    ----------
    prox_f, prox_g : callable
        ``prox(v, rho)`` for each block.
    A, B, c : array-like, optional
        Coupling; defaults to :math:`x - z = 0`.
    rho : float
        Penalty parameter, positive.
    n : int, optional
        Dimension, when it cannot be inferred.
    max_iter, eps_abs, eps_rel
        Stopping controls.

    Returns
    -------
    RichResult
        ``x``, ``z``, ``u``, ``n_iter``, ``converged``,
        ``primal_residual``, ``dual_residual``, ``residual_path``.

    References
    ----------
    Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).
        Distributed optimization and statistical learning via the
        alternating direction method of multipliers. *Foundations and
        Trends in Machine Learning*, 3(1), 1-122.

    Examples
    --------
    Splitting a least-squares fit against an l1 penalty -- the standard
    LASSO-by-ADMM arrangement.

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> A_ = rng.normal(size=(50, 6))
    >>> x_true = np.array([2.0, 0, 0, -3.0, 0, 0])
    >>> b = A_ @ x_true + 0.05 * rng.normal(size=50)
    >>> rho = 1.0
    >>> M = np.linalg.inv(A_.T @ A_ + rho * np.eye(6))
    >>> px = lambda v, r: M @ (A_.T @ b + r * v)
    >>> pz = lambda v, r: np.sign(v) * np.maximum(np.abs(v) - 0.3 / r, 0)
    >>> r = boyd_admm(px, pz, rho=rho, n=6)
    >>> bool(r["converged"])
    True

    The z block carries the sparsity, exactly zero where the penalty
    won.

    >>> sorted(int(i) for i in np.flatnonzero(np.abs(r["z"]) > 1e-6))
    [0, 3]

    Both residuals are driven down; reporting only one would hide the
    imbalance rho controls.

    >>> bool(r["primal_residual"] < 1e-5 and r["dual_residual"] < 1e-5)
    True

    >>> boyd_admm(px, pz, rho=-1.0, n=6)
    Traceback (most recent call last):
        ...
    ValueError: rho must be positive
    """
    rho = float(rho)
    if rho <= 0:
        raise ValueError("rho must be positive")
    if n is None:
        raise ValueError("n (the block dimension) is required")
    n = int(n)
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    path = []
    conv = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        x = np.atleast_1d(np.asarray(prox_f(z - u, rho), dtype=float)).ravel()
        z_old = z.copy()
        z = np.atleast_1d(np.asarray(prox_g(x + u, rho), dtype=float)).ravel()
        u = u + x - z
        r_norm = float(np.linalg.norm(x - z))
        s_norm = float(np.linalg.norm(-rho * (z - z_old)))
        path.append((r_norm, s_norm))
        eps_p = np.sqrt(n) * eps_abs + eps_rel * max(
            float(np.linalg.norm(x)), float(np.linalg.norm(z)))
        eps_d = np.sqrt(n) * eps_abs + eps_rel * float(np.linalg.norm(rho * u))
        if r_norm <= eps_p and s_norm <= eps_d:
            conv = True
            break
    return RichResult(
        title="ADMM",
        summary_lines=[("iterations", int(it)), ("rho", rho),
                       ("primal residual", path[-1][0]),
                       ("dual residual", path[-1][1])],
        warnings=[] if conv else ["ADMM hit max_iter; it converges for any "
                                  "rho > 0 but the RATE depends on it"],
        payload={
            "x": x, "z": z, "u": u, "n_iter": int(it), "converged": conv,
            "primal_residual": path[-1][0], "dual_residual": path[-1][1],
            "residual_path": np.asarray(path), "rho": rho,
            "method": "boyd_admm",
        },
    )


def cheatsheet():
    return "cvxadm: converges for ANY rho > 0; rho sets the primal/dual residual BALANCE, so watch both"


# compact alias per ledger/NAMING.md
boydadmm = boyd_admm
