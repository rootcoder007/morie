# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Network SI epidemic (individual-based mean field)."""

from __future__ import annotations

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["si_epidemic"]


def _adj(G):
    A = [[float(v) for v in row] for row in np.atleast_2d(np.asarray(G, dtype=float)).tolist()]
    n = len(A)
    if n == 0 or any(len(r) != n for r in A):
        raise ValueError("si_epidemic: G must be a square adjacency matrix")
    return A, n


def _p0(initial, n):
    p = [float(v) for v in np.atleast_1d(np.asarray(initial, dtype=float)).tolist()]
    if len(p) != n:
        raise ValueError("si_epidemic: initial must have one entry per node")
    if any(v < 0.0 or v > 1.0 for v in p):
        raise ValueError("si_epidemic: initial probabilities must lie in [0, 1]")
    return p


def si_epidemic(G, beta, initial, t_max=20.0, dt=0.01):
    r"""Susceptible-Infected spreading on a fixed network, individual based.

    Each node carries a probability :math:`\rho_i(t)` of being infected and
    obeys the mean-field reaction rate equation

    .. math::

        \partial_t \rho_i(t) = \beta \, [1 - \rho_i(t)] \sum_j A_{ij}\,\rho_j(t)

    which is Pastor-Satorras & Vespignani's eq. (7) written per node rather
    than per degree class: writing :math:`k_i = \sum_j A_{ij}` and
    :math:`\Theta_i = k_i^{-1}\sum_j A_{ij}\rho_j` for the probability that a
    link out of node *i* points at an infected node, the right-hand side is
    :math:`\beta k_i [1 - \rho_i]\Theta_i`.  Dropping the recovery term
    :math:`-\rho_i` of eq. (7) gives SI rather than SIS.  Under the annealed
    (degree-block) approximation, in which all nodes of degree *k* are
    equivalent, this reduces exactly to eq. (7) without recovery.

    Integrated with classical fourth-order Runge-Kutta on a fixed step, so the
    result is deterministic and identical in every language arm.

    Parameters
    ----------
    G : array-like
        Square adjacency matrix, one row/column per node.  Weights are used
        as given; symmetry is not required.
    beta : float
        Per-edge infection rate.
    initial : array-like
        Initial infection probability of each node, in [0, 1].
    t_max : float, default 20.0
        Integration horizon.
    dt : float, default 0.01
        Runge-Kutta step.

    Returns
    -------
    RichResult
        ``estimate`` is the final prevalence :math:`\rho(t_{max})`, the mean
        of the node probabilities (eq. (10) with the empirical degree
        distribution).  ``theta`` is eq. (9) evaluated on the final state.

    References
    ----------
    Pastor-Satorras, R. & Vespignani, A. (2001). Epidemic dynamics and
    endemic states in complex networks. Physical Review E 63, 066117,
    eqs. (7)-(10). doi:10.1103/PhysRevE.63.066117
    """
    A, n = _adj(G)
    p = _p0(initial, n)
    beta = float(beta)
    t_max = float(t_max)
    dt = float(dt)
    if beta < 0.0:
        raise ValueError("si_epidemic: beta must be non-negative")
    if dt <= 0.0 or t_max < 0.0:
        raise ValueError("si_epidemic: need dt > 0 and t_max >= 0")

    def deriv(x):
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i][j] * x[j]
            out[i] = beta * (1.0 - x[i]) * s
        return out

    nsteps = int(round(t_max / dt))
    rho0 = sum(p) / n
    half_time = float("nan")
    prev_rho = rho0
    for step in range(nsteps):
        k1 = deriv(p)
        k2 = deriv([p[i] + 0.5 * dt * k1[i] for i in range(n)])
        k3 = deriv([p[i] + 0.5 * dt * k2[i] for i in range(n)])
        k4 = deriv([p[i] + dt * k3[i] for i in range(n)])
        p = [p[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(n)]
        rho = sum(p) / n
        if prev_rho < 0.5 <= rho:
            half_time = (step + 1) * dt
        prev_rho = rho

    deg = [sum(A[i]) for i in range(n)]
    kbar = sum(deg) / n
    k2bar = sum(d * d for d in deg) / n
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
            "half_time": half_time,
            "n": n,
            "beta": beta,
            "t_max": t_max,
            "dt": dt,
            "method": "Network SI epidemic, individual-based mean field (Pastor-Satorras & Vespignani 2001, eq. 7)",
        }
    )


def cheatsheet():
    return "siepid: Network SI epidemic (PSV 2001 eq. 7 without recovery)"


# compact alias per ledger/NAMING.md
siepidemic = si_epidemic
