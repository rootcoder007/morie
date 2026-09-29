"""Spatial SIR diffusion"""

from ._containers import SpatialResult


def _rhs(S, Inf, N, C, beta, gamma):
    n = len(S)
    force = [sum(C[i][j] * Inf[j] / N[j] for j in range(n)) for i in range(n)]
    dS = [-beta * S[i] * force[i] for i in range(n)]
    dI = [beta * S[i] * force[i] - gamma * Inf[i] for i in range(n)]
    dR = [gamma * Inf[i] for i in range(n)]
    return dS, dI, dR


def spatial_sir(data, W=None, N=None, *, beta=0.5, gamma=0.2, kappa=0.1, t_max=100.0, dt=0.1, method="rk4"):
    r"""Metapopulation (spatially coupled) SIR epidemic integrated by fourth-order Runge-Kutta.

    Patch ``i`` has ``S_i + I_i + R_i = N_i``; infection pressure mixes the
    local prevalence with that of the neighbours through the coupling
    ``C = (1 - kappa) I + kappa W~`` (``W~`` the row-standardised weights):

    ``dS_i/dt = -beta S_i sum_j C_ij I_j / N_j``,
    ``dI_i/dt = beta S_i sum_j C_ij I_j / N_j - gamma I_i``,
    ``dR_i/dt = gamma I_i``

    (Keeling and Rohani 2008, sec. 7.3). ``data`` holds the initially
    infected counts per patch, ``N`` the populations (default 1000 each)
    and ``W`` the adjacency (default: none, i.e. isolated patches). Returns
    the trajectories on the ``dt`` grid; ``statistic`` is the final
    attack rate ``sum R(t_max) / sum N``.

    References
    ----------
    Keeling, M. J. and Rohani, P. (2008). *Modeling Infectious Diseases in
    Humans and Animals*. Princeton University Press.

    Examples
    --------
    >>> W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> r = spatial_sir([10, 0, 0], W, [1000, 1000, 1000], beta=0.6, gamma=0.2, t_max=60, dt=0.5)
    >>> round(r.statistic, 8)
    0.93992343
    """
    if method != "rk4":
        raise ValueError("method must be 'rk4'")
    I0 = [float(v) for v in (data.tolist() if hasattr(data, "tolist") else data)]
    n = len(I0)
    Nv = [1000.0] * n if N is None else [float(v) for v in N]
    A = [[0.0] * n for _ in range(n)] if W is None else [[float(v) for v in r] for r in W]
    Wt = [[v / sum(r) if sum(r) > 0 else 0.0 for v in r] for r in A]
    C = [[(1.0 - kappa) * (i == j) + kappa * Wt[i][j] for j in range(n)] for i in range(n)]
    S = [Nv[i] - I0[i] for i in range(n)]
    Inf = list(I0)
    R = [0.0] * n
    steps = int(round(t_max / dt))
    times, traj = [0.0], [(list(S), list(Inf), list(R))]
    for k in range(steps):
        k1 = _rhs(S, Inf, Nv, C, beta, gamma)
        s2 = [[x + 0.5 * dt * d for x, d in zip(v, kk)] for v, kk in zip((S, Inf, R), k1)]
        k2 = _rhs(s2[0], s2[1], Nv, C, beta, gamma)
        s3 = [[x + 0.5 * dt * d for x, d in zip(v, kk)] for v, kk in zip((S, Inf, R), k2)]
        k3 = _rhs(s3[0], s3[1], Nv, C, beta, gamma)
        s4 = [[x + dt * d for x, d in zip(v, kk)] for v, kk in zip((S, Inf, R), k3)]
        k4 = _rhs(s4[0], s4[1], Nv, C, beta, gamma)
        S, Inf, R = (
            [x + dt / 6.0 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(v, k1[m], k2[m], k3[m], k4[m])]
            for m, v in enumerate((S, Inf, R))
        )
        times.append((k + 1) * dt)
        traj.append((list(S), list(Inf), list(R)))
    attack = sum(R) / sum(Nv)
    return SpatialResult(
        name="zesir",
        statistic=attack,
        extra={
            "times": times,
            "S": [t[0] for t in traj],
            "I": [t[1] for t in traj],
            "R": [t[2] for t in traj],
            "attack_rate": attack,
            "R0_local": beta / gamma,
        },
    )


spat = spatial_sir


def cheatsheet() -> str:
    return "spatial_sir(I0, W, N, beta, gamma, kappa) -> metapopulation SIR by RK4 (Keeling and Rohani 2008)."


# compact alias per ledger/NAMING.md
spatialsir = spatial_sir
