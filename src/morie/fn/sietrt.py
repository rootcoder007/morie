# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Network SIS epidemic (individual-based mean field)."""

from __future__ import annotations

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["sis_epidemic"]


def _adj(G):
    A = [[float(v) for v in row] for row in np.atleast_2d(np.asarray(G, dtype=float)).tolist()]
    n = len(A)
    if n == 0 or any(len(r) != n for r in A):
        raise ValueError("sis_epidemic: G must be a square adjacency matrix")
    return A, n


def sis_epidemic(G, beta, gamma, initial, t_max=50.0, dt=0.01):
    r"""Susceptible-Infected-Susceptible spreading on a fixed network.

    Individual-based form of Pastor-Satorras & Vespignani eq. (7),

    .. math::

        \partial_t \rho_i(t) = -\gamma\,\rho_i(t)
            + \beta\,[1 - \rho_i(t)]\sum_j A_{ij}\,\rho_j(t),

    the recovered node returning directly to the susceptible pool.  Writing
    :math:`k_i=\sum_j A_{ij}` and :math:`\Theta_i=k_i^{-1}\sum_j A_{ij}\rho_j`
    the creation term is :math:`\beta k_i[1-\rho_i]\Theta_i`, which is eq. (7)
    verbatim; PSV set :math:`\gamma = 1` (unitary recovery rate) so that only
    the ratio :math:`\lambda = \beta/\gamma` matters.

    The reported ``lambda_c`` is the heterogeneous mean-field epidemic
    threshold :math:`\lambda_c = \langle k\rangle/\langle k^2\rangle` obtained
    by linearising eqs. (8)-(9) about :math:`\Theta = 0`; the endemic phase is
    :math:`\lambda > \lambda_c`.  For a k-regular graph this is
    :math:`1/k`, and PSV's homogeneous result :math:`\langle
    k\rangle\lambda_c = 1` (their eq. (5) discussion) follows.

    Parameters
    ----------
    G : array-like
        Square adjacency matrix.
    beta : float
        Per-edge infection rate.
    gamma : float
        Recovery rate back to susceptible.
    initial : array-like
        Initial infection probability of each node, in [0, 1].
    t_max : float, default 50.0
        Integration horizon.
    dt : float, default 0.01
        Runge-Kutta step.

    Returns
    -------
    RichResult
        ``estimate`` is the prevalence :math:`\rho(t_{max})` (eq. (10)).

    References
    ----------
    Pastor-Satorras, R. & Vespignani, A. (2001). Epidemic dynamics and
    endemic states in complex networks. Physical Review E 63, 066117,
    eqs. (7)-(10). doi:10.1103/PhysRevE.63.066117
    """
    A, n = _adj(G)
    p = [float(v) for v in np.atleast_1d(np.asarray(initial, dtype=float)).tolist()]
    if len(p) != n:
        raise ValueError("sis_epidemic: initial must have one entry per node")
    if any(v < 0.0 or v > 1.0 for v in p):
        raise ValueError("sis_epidemic: initial probabilities must lie in [0, 1]")
    beta = float(beta)
    gamma = float(gamma)
    t_max = float(t_max)
    dt = float(dt)
    if beta < 0.0 or gamma < 0.0:
        raise ValueError("sis_epidemic: beta and gamma must be non-negative")
    if dt <= 0.0 or t_max < 0.0:
        raise ValueError("sis_epidemic: need dt > 0 and t_max >= 0")

    def deriv(x):
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i][j] * x[j]
            out[i] = -gamma * x[i] + beta * (1.0 - x[i]) * s
        return out

    rho0 = sum(p) / n
    nsteps = int(round(t_max / dt))
    for _ in range(nsteps):
        k1 = deriv(p)
        k2 = deriv([p[i] + 0.5 * dt * k1[i] for i in range(n)])
        k3 = deriv([p[i] + 0.5 * dt * k2[i] for i in range(n)])
        k4 = deriv([p[i] + dt * k3[i] for i in range(n)])
        p = [p[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(n)]

    deg = [sum(A[i]) for i in range(n)]
    kbar = sum(deg) / n
    k2bar = sum(d * d for d in deg) / n
    lam = beta / gamma if gamma > 0.0 else float("inf")
    lam_c = kbar / k2bar if k2bar > 0.0 else float("nan")
    theta = (sum(deg[i] * p[i] for i in range(n)) / (n * kbar)) if kbar > 0.0 else float("nan")
    rho = sum(p) / n

    return RichResult(
        payload={
            "estimate": rho,
            "prevalence": p,
            "rho_final": rho,
            "rho_initial": rho0,
            "theta": theta,
            "mean_degree": kbar,
            "second_moment": k2bar,
            "lambda": lam,
            "lambda_c": lam_c,
            "endemic": 1.0 if lam > lam_c else 0.0,
            "n": n,
            "beta": beta,
            "gamma": gamma,
            "t_max": t_max,
            "dt": dt,
            "method": "Network SIS epidemic, individual-based mean field (Pastor-Satorras & Vespignani 2001, eq. 7)",
        }
    )


def cheatsheet():
    return "sietrt: Network SIS epidemic (PSV 2001 eq. 7)"


# compact alias per ledger/NAMING.md
sisepidemic = sis_epidemic
