# morie.fn -- function file (rootcoder007/morie)
"""Proximal gradient -- Boyd & Vandenberghe / Parikh & Boyd (2014)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_proximal_grad"]


def boyd_proximal_grad(f, grad_f, prox, x0, t=0.1, max_iter=500,
                       tol=1e-09, h=None):
    r"""Iterate :math:`x^{k+1} = \operatorname{prox}_{t h}\!\left(x^k -
    t\nabla f(x^k)\right)` for :math:`\min f(x) + h(x)`.

    The split is the point: f must be smooth, h need only have a
    computable proximal operator. That is what lets a NONSMOOTH penalty
    like :math:`\lVert x\rVert_1` be handled at the speed of gradient
    descent -- its prox is soft thresholding, in closed form -- instead of
    by a subgradient method at :math:`O(1/\sqrt k)`.

    The rate is :math:`O(1/k)`, the same as gradient descent on the smooth
    part alone: the nonsmooth term costs nothing asymptotically, which is
    the result that makes the whole family worth using.

    Parameters
    ----------
    f, grad_f : callable
        Smooth part and its gradient.
    prox : callable
        ``prox(v, t)`` returning the proximal operator of ``t*h``.
    x0 : array-like
        Start.
    t : float
        Step size; needs :math:`t \le 1/L`.
    max_iter, tol
        Stopping controls.
    h : callable, optional
        The nonsmooth part, for reporting the full objective.

    Returns
    -------
    RichResult
        ``x``, ``f``, ``objective``, ``n_iter``, ``converged``,
        ``n_zero`` (exactly-zero entries), ``objective_path``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.
    Parikh, N., & Boyd, S. (2014). Proximal algorithms. *Foundations
        and Trends in Optimization*, 1(3), 123-231.

    Examples
    --------
    LASSO by proximal gradient: soft thresholding is the prox of the
    l1 norm.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> A = rng.normal(size=(60, 8))
    >>> x_true = np.array([3.0, 0, 0, -2.0, 0, 0, 0, 0])
    >>> b = A @ x_true + 0.05 * rng.normal(size=60)
    >>> soft = lambda v, s: np.sign(v) * np.maximum(np.abs(v) - s, 0.0)
    >>> lam = 5.0
    >>> r = boyd_proximal_grad(lambda z: 0.5 * np.sum((A @ z - b) ** 2),
    ...                        lambda z: A.T @ (A @ z - b),
    ...                        lambda v, s: soft(v, s * lam),
    ...                        np.zeros(8), t=1 / np.linalg.norm(A, 2) ** 2,
    ...                        h=lambda z: lam * np.abs(z).sum())

    The solution is genuinely SPARSE -- exact zeros, not small numbers,
    which is what the prox buys and a smooth penalty cannot deliver.

    >>> int(r["n_zero"])
    6

    Lambda has to clear the noise scale for that to happen. At lam = 0.5
    on this data the same solver leaves the irrelevant coefficients at
    around 2e-3: nonzero, small, and indistinguishable from a ridge fit.
    Sparsity is a property of the penalty strength, not of the algorithm.

    >>> weak = boyd_proximal_grad(lambda z: 0.5 * np.sum((A @ z - b) ** 2),
    ...                           lambda z: A.T @ (A @ z - b),
    ...                           lambda v, s: soft(v, s * 0.5),
    ...                           np.zeros(8), t=1 / np.linalg.norm(A, 2) ** 2)
    >>> int(weak["n_zero"])
    3

    The nonzero entries land on the true support.

    >>> sorted(int(i) for i in np.flatnonzero(np.abs(r["x"]) > 1e-8))
    [0, 3]

    The objective decreases monotonically at a valid step size.

    >>> bool(np.all(np.diff(r["objective_path"]) <= 1e-9))
    True
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).ravel().copy()
    obj = [float(f(x)) + (float(h(x)) if h is not None else 0.0)]
    conv = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        g = np.atleast_1d(np.asarray(grad_f(x), dtype=float)).ravel()
        x_new = np.atleast_1d(np.asarray(prox(x - t * g, t),
                                         dtype=float)).ravel()
        obj.append(float(f(x_new)) + (float(h(x_new)) if h is not None else 0.0))
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            conv = True
            break
        x = x_new
    return RichResult(
        title="Proximal gradient",
        summary_lines=[("iterations", int(it)), ("objective", obj[-1]),
                       ("converged", conv),
                       ("exact zeros", int(np.sum(np.abs(x) <= 1e-12)))],
        payload={
            "x": x, "f": float(f(x)), "objective": obj[-1],
            "n_iter": int(it), "converged": conv,
            "n_zero": int(np.sum(np.abs(x) <= 1e-12)),
            "objective_path": np.asarray(obj), "step": float(t),
            "method": "boyd_proximal_grad",
        },
    )


def cheatsheet():
    return "cvxprg: nonsmooth term is FREE asymptotically -- O(1/k), same as smooth gradient descent"
