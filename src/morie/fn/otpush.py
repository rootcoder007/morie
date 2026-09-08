"""Pushforward density via change-of-variables and Jacobian."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["ot_pushforward_density"]


def ot_pushforward_density(mu_grid, T_jac, T_inv_grid):
    """
    Push-forward of a density through a smooth bijective map.

    Formula: nu(y) = mu(T^{-1}(y)) / |det DT(T^{-1}(y))|

    Verified against Peyre & Cuturi (2019), Remark 2.6, eq. (2.8) --
    source consulted, which writes ``rho_alpha(x) = |det(T'(x))|
    rho_beta(T(x))``; solving for ``rho_beta`` gives the form used here.

    Parameters
    ----------
    mu_grid : array-like
        Source density evaluated at ``T^{-1}(y)`` for each output point.
    T_jac : array-like
        Jacobian determinant of ``T`` at those same preimages.
    T_inv_grid : array-like
        The preimages themselves; carried through to the result so the
        caller can see which point each value belongs to.

    Returns
    -------
    RichResult
        Keys: estimate (the push-forward density), preimage, jacobian,
        n, method.

    References
    ----------
    Peyre, G. & Cuturi, M. (2019). Computational Optimal Transport,
    Remark 2.6, eq. (2.8).
    """
    mu = [float(t) for t in np.atleast_1d(np.asarray(mu_grid, dtype=float))]
    jac = [float(t) for t in np.atleast_1d(np.asarray(T_jac, dtype=float))]
    pre = [float(t) for t in np.atleast_1d(np.asarray(T_inv_grid, dtype=float))]
    n = len(mu)
    if len(jac) != n or len(pre) != n:
        raise ValueError("mu_grid, T_jac and T_inv_grid must have the same length")
    if min(mu) < 0.0:
        raise ValueError("mu_grid must be non-negative")
    out = []
    for i in range(n):
        d = abs(jac[i])
        if d <= 0.0:
            raise ValueError("the map is singular: |det DT| = 0")
        out.append(mu[i] / d)
    return RichResult(
        payload={
            "estimate": out,
            "preimage": pre,
            "jacobian": jac,
            "n": n,
            "method": "Push-forward density -- Peyre & Cuturi (2019) eq. (2.8)",
        }
    )


def cheatsheet():
    return "otpush: Pushforward density via change-of-variables and Jacobian"
