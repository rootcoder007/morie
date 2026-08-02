# morie.fn -- function file (rootcoder007/morie)
"""Newton decrement -- Boyd & Vandenberghe (2004) Sec 9.5.1."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_newton_decrement"]


def boyd_newton_decrement(grad, hess):
    r"""The Newton decrement and the suboptimality bound it certifies.

    .. math::
        \lambda(x) = \left(\nabla f(x)^\top \nabla^2 f(x)^{-1}
                     \nabla f(x)\right)^{1/2}.

    Its value is that :math:`\lambda^2/2` estimates :math:`f(x) - p^\star`,
    exactly so for a quadratic, which makes :math:`\lambda^2/2 \le \varepsilon`
    the standard stopping rule for Newton's method. Unlike
    :math:`\lVert \nabla f \rVert` it is affine-invariant: rescaling the
    variables leaves it unchanged, so the same tolerance means the same thing
    across parameterisations.

    Parameters
    ----------
    grad : array-like
        Gradient at ``x``, shape ``(n,)``.
    hess : array-like
        Hessian at ``x``, shape ``(n, n)``, symmetric positive definite.

    Returns
    -------
    RichResult
        ``decrement`` :math:`\lambda`, ``suboptimality`` :math:`\lambda^2/2`,
        and ``grad_norm`` for comparison.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    For ``f(x) = 0.5 x'Ax`` the bound is exact.

    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> x = np.array([5.0, -2.0])
    >>> r = boyd_newton_decrement(A @ x, A)
    >>> f = 0.5 * x @ A @ x
    >>> bool(abs(r["suboptimality"] - f) < 1e-10)
    True

    Affine invariance -- rescaling the variables does not move it.

    >>> S = np.diag([10.0, 0.1])
    >>> r2 = boyd_newton_decrement(S @ (A @ x), S @ A @ S)
    >>> bool(abs(r2["decrement"] - r["decrement"]) < 1e-9)
    True
    """
    g = np.atleast_1d(np.asarray(grad, dtype=float)).ravel()
    Hm = np.atleast_2d(np.asarray(hess, dtype=float))
    n = g.size
    if Hm.shape != (n, n):
        raise ValueError(f"hess must be ({n}, {n}) to match grad, got {Hm.shape}")
    try:
        quad = float(g @ np.linalg.solve(Hm, g))
    except np.linalg.LinAlgError as exc:
        raise ValueError("hess is singular; the Newton decrement is undefined") from exc
    if quad < 0:
        raise ValueError(
            "grad' H^-1 grad is negative, so hess is not positive definite and "
            "the decrement is not defined"
        )
    lam = float(np.sqrt(quad))
    return RichResult(
        title="Newton decrement",
        summary_lines=[("lambda", lam), ("lambda^2/2", lam**2 / 2)],
        payload={
            "decrement": lam,
            "suboptimality": lam**2 / 2,
            "grad_norm": float(np.linalg.norm(g)),
            "method": "boyd_newton_decrement",
        },
    )


def cheatsheet():
    return "cvxnda: lambda^2/2 estimates f(x)-p*; affine-invariant, unlike ||grad||"
