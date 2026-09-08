# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Network SIR epidemic (individual-based mean field)."""

from __future__ import annotations

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["sir_epidemic"]


def sir_epidemic(G, beta, gamma, initial, t_max=50.0, dt=0.01):
    r"""Susceptible-Infected-Recovered spreading on a fixed network.

    The individual-based mean-field system, obtained from
    Pastor-Satorras & Vespignani's eq. (7) by adding an absorbing recovered
    compartment instead of returning recovered nodes to the susceptible pool:

    .. math::

        \partial_t S_i = -\beta\, S_i \sum_j A_{ij} I_j, \qquad
        \partial_t I_i = \beta\, S_i \sum_j A_{ij} I_j - \gamma I_i, \qquad
        \partial_t R_i = \gamma I_i .

    With :math:`\gamma = 0` the R compartment never fills and the system is
    exactly the SI model of :mod:`morie.fn.siepid`.  The per-node total
    :math:`S_i + I_i + R_i` is conserved by construction, and is reported as
    ``conservation_error`` so the integration can be checked.

    Parameters
    ----------
    G : array-like
        Square adjacency matrix.
    beta : float
        Per-edge infection rate.
    gamma : float
        Recovery rate into the absorbing R compartment.
    initial : array-like
        Initial infection probability of each node, in [0, 1]; the
        complement starts susceptible.
    t_max : float, default 50.0
        Integration horizon.
    dt : float, default 0.01
        Runge-Kutta step.

    Returns
    -------
    RichResult
        ``estimate`` is the final attack rate, the mean of :math:`R_i`.

    References
    ----------
    Pastor-Satorras, R. & Vespignani, A. (2001). Epidemic dynamics and
    endemic states in complex networks. Physical Review E 63, 066117,
    eq. (7). doi:10.1103/PhysRevE.63.066117
    """
    A = [[float(v) for v in row] for row in np.atleast_2d(np.asarray(G, dtype=float)).tolist()]
    n = len(A)
    if n == 0 or any(len(r) != n for r in A):
        raise ValueError("sir_epidemic: G must be a square adjacency matrix")
    I = [float(v) for v in np.atleast_1d(np.asarray(initial, dtype=float)).tolist()]
    if len(I) != n:
        raise ValueError("sir_epidemic: initial must have one entry per node")
    if any(v < 0.0 or v > 1.0 for v in I):
        raise ValueError("sir_epidemic: initial probabilities must lie in [0, 1]")
    beta = float(beta)
    gamma = float(gamma)
    t_max = float(t_max)
    dt = float(dt)
    if beta < 0.0 or gamma < 0.0:
        raise ValueError("sir_epidemic: beta and gamma must be non-negative")
    if dt <= 0.0 or t_max < 0.0:
        raise ValueError("sir_epidemic: need dt > 0 and t_max >= 0")

    S = [1.0 - v for v in I]
    R = [0.0] * n

    def deriv(s, i, r):
        ds = [0.0] * n
        di = [0.0] * n
        dr = [0.0] * n
        for a in range(n):
            f = 0.0
            for b in range(n):
                f += A[a][b] * i[b]
            f *= beta * s[a]
            ds[a] = -f
            di[a] = f - gamma * i[a]
            dr[a] = gamma * i[a]
        return ds, di, dr

    nsteps = int(round(t_max / dt))
    peak_I = sum(I) / n
    peak_time = 0.0
    for step in range(nsteps):
        a1, b1, c1 = deriv(S, I, R)
        a2, b2, c2 = deriv(
            [S[i] + 0.5 * dt * a1[i] for i in range(n)],
            [I[i] + 0.5 * dt * b1[i] for i in range(n)],
            [R[i] + 0.5 * dt * c1[i] for i in range(n)],
        )
        a3, b3, c3 = deriv(
            [S[i] + 0.5 * dt * a2[i] for i in range(n)],
            [I[i] + 0.5 * dt * b2[i] for i in range(n)],
            [R[i] + 0.5 * dt * c2[i] for i in range(n)],
        )
        a4, b4, c4 = deriv(
            [S[i] + dt * a3[i] for i in range(n)],
            [I[i] + dt * b3[i] for i in range(n)],
            [R[i] + dt * c3[i] for i in range(n)],
        )
        S = [S[i] + (dt / 6.0) * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]) for i in range(n)]
        I = [I[i] + (dt / 6.0) * (b1[i] + 2.0 * b2[i] + 2.0 * b3[i] + b4[i]) for i in range(n)]
        R = [R[i] + (dt / 6.0) * (c1[i] + 2.0 * c2[i] + 2.0 * c3[i] + c4[i]) for i in range(n)]
        cur = sum(I) / n
        if cur > peak_I:
            peak_I = cur
            peak_time = (step + 1) * dt

    deg = [sum(A[i]) for i in range(n)]
    kbar = sum(deg) / n
    cons = max(abs(S[i] + I[i] + R[i] - 1.0) for i in range(n))

    return RichResult(
        payload={
            "estimate": sum(R) / n,
            "S": S,
            "I": I,
            "R": R,
            "attack_rate": sum(R) / n,
            "prevalence": sum(I) / n,
            "susceptible": sum(S) / n,
            "peak_prevalence": peak_I,
            "peak_time": peak_time,
            "mean_degree": kbar,
            "conservation_error": cons,
            "n": n,
            "beta": beta,
            "gamma": gamma,
            "t_max": t_max,
            "dt": dt,
            "method": "Network SIR epidemic, individual-based mean field (Pastor-Satorras & Vespignani 2001, eq. 7)",
        }
    )


def cheatsheet():
    return "siaepi: Network SIR epidemic (PSV 2001 eq. 7 with absorbing R)"


# compact alias per ledger/NAMING.md
sirepidemic = sir_epidemic
