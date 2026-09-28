# morie.fn -- function file (rootcoder007/morie)
"""Police operations research: the Larson hypercube queueing model of patrol units, the Kolesar-Blum square-root
law for response distance, and the Short et al. residential-burglary hotspot model (agent lattice and
reaction-diffusion PDE with hotspot suppression)."""

from __future__ import annotations

import math

from ._qpcore import solve, ssum
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["hypercube_queue", "square_root_law", "short_crime_lattice", "short_crime_pde"]


def hypercube_queue(
    arrival_rates, preferences, *, service_rate: float = 1.0, queue: str = "zero", travel_time=None
) -> RichResult:
    r"""Larson (1974) hypercube queueing model of ``N`` distinguishable patrol units serving ``J`` atoms.

    States are the ``2^N`` busy/free patterns. A call from atom ``j``
    (Poisson rate ``lambda_j``) is dispatched to the first free unit in
    ``preferences[j]``; each busy unit completes service at rate ``mu``. With
    ``queue="zero"`` calls arriving when every unit is busy are lost (or
    handled elsewhere); with ``queue="infinite"`` they wait FCFS: the queue
    states above "all busy" are aggregated using their geometric law, so only
    ``P(all busy, empty queue) = (1 - rho) P(all busy)``, ``rho = Lambda/(N
    mu)``, feeds service completions back into the cube. The balance
    equations ``pi Q = 0`` are solved exactly.

    Returned: state probabilities (index = bit pattern, bit ``n`` = unit
    ``n`` busy), unit workloads, the fraction of all calls from atom ``j``
    handled by unit ``n`` (``dispatch``), the loss (zero line) or waiting
    (infinite line) probability, the mean queue length (infinite line) and,
    with ``travel_time`` (units x atoms), the mean travel time of dispatched
    calls and of each atom.

    References
    ----------
    Larson, R. C. (1974). A hypercube queuing model for facility location
    and redistricting in urban emergency services. *Computers and
    Operations Research*, 1(1), 67-95.
    Larson, R. C. and Odoni, A. R. (1981). *Urban Operations Research*.
    Prentice-Hall, chapter 5.

    Examples
    --------
    >>> r = hypercube_queue([1.0], [[0, 1]], service_rate=1.0)
    >>> round(r.loss, 6), [round(v, 6) for v in r.workload]
    (0.2, [0.5, 0.3])
    """
    lam = [float(v) for v in arrival_rates]
    pref = [list(p) for p in preferences]
    N = max(max(p) for p in pref) + 1
    if N > 10:
        raise ValueError("at most 10 units (2^N states)")
    mu = float(service_rate)
    S = 1 << N
    full = S - 1
    Lam = ssum(lam)
    rho = Lam / (N * mu)
    if queue == "infinite" and rho >= 1:
        raise ValueError("infinite-line model needs Lambda < N mu")
    Q = [[0.0] * S for _ in range(S)]
    for b in range(S):
        if b != full:
            for j, lj in enumerate(lam):
                u = next(u for u in pref[j] if not (b >> u) & 1)
                Q[b][b | (1 << u)] += lj
        scale = (1 - rho) if (b == full and queue == "infinite") else 1.0
        for i in range(N):
            if (b >> i) & 1:
                Q[b][b & ~(1 << i)] += mu * scale
        Q[b][b] = -ssum(Q[b][k] for k in range(S) if k != b)
    # solve pi Q = 0 with sum pi = 1: replace the last balance equation by the normalisation
    A = [[Q[k][r] for k in range(S)] for r in range(S)]
    A[-1] = [1.0] * S
    rhs = [0.0] * (S - 1) + [1.0]
    pi = [float(v) for v in solve(A, rhs)]
    work = [ssum(pi[b] for b in range(S) if (b >> n) & 1) for n in range(N)]
    disp = [[0.0] * len(lam) for _ in range(N)]
    for b in range(S):
        if b == full:
            continue
        for j, lj in enumerate(lam):
            u = next(u for u in pref[j] if not (b >> u) & 1)
            disp[u][j] += lj * pi[b] / Lam
    out = {"state_prob": pi, "workload": work, "dispatch": disp, "loss" if queue == "zero" else "wait": pi[full]}
    if queue == "infinite":
        out["mean_queue"] = pi[full] * rho / (1 - rho)
    if travel_time is not None:
        T = [[float(v) for v in r] for r in travel_time]
        served = ssum(ssum(r) for r in disp)
        out["mean_travel_time"] = ssum(disp[n][j] * T[n][j] for n in range(N) for j in range(len(lam))) / served
        out["atom_travel_time"] = [
            ssum(disp[n][j] * T[n][j] for n in range(N)) / ssum(disp[n][j] for n in range(N)) for j in range(len(lam))
        ]
    return RichResult(payload=out)


def square_root_law(
    area: float, available_units: float, *, metric: str = "euclidean", speed: float | None = None
) -> RichResult:
    r"""Kolesar and Blum (1973) square-root law: mean distance to the nearest available unit ``c sqrt(A / n)``.

    For units spread uniformly (Poisson) over area ``A`` with ``n``
    available, the nearest-unit distance has mean ``1/(2 sqrt(n/A))``
    (Euclidean, ``c = 0.5``) or ``sqrt(2 pi)/4 / sqrt(n/A)`` (right-angle
    travel, ``c = 0.6267``). With ``speed`` the mean travel time is returned
    as well. Use ``n = N (1 - rho)`` for ``N`` units busy a fraction ``rho``
    of the time.

    References
    ----------
    Kolesar, P. and Blum, E. H. (1973). Square root laws for fire engine
    response distances. *Management Science*, 19(12), 1368-1378.

    Examples
    --------
    >>> round(square_root_law(100.0, 4.0).distance, 6)
    2.5
    """
    if metric == "euclidean":
        c = 0.5
    elif metric == "rectilinear":
        c = math.sqrt(2 * math.pi) / 4
    else:
        raise ValueError("metric must be euclidean or rectilinear")
    d = c * math.sqrt(area / available_units)
    out = {"distance": d, "constant": c}
    if speed is not None:
        out["time"] = d / speed
    return RichResult(payload=out)


def _poisson_inv(u, lam):
    k, p = 0, math.exp(-lam)
    c = p
    while u > c and k < 1000:
        k += 1
        p *= lam / k
        c += p
    return k


def short_crime_lattice(
    n: int,
    steps: int,
    *,
    A0: float = 1 / 30,
    eta: float = 0.03,
    omega: float = 1 / 15,
    gamma: float = 0.002,
    B0=None,
    criminals=None,
    seed: int = 1,
) -> RichResult:
    r"""Short et al. (2008) agent-based residential burglary model on an ``n`` x ``n`` periodic lattice.

    Attractiveness ``A_s = A0 + B_s``. Each step: every criminal (in list
    order) burgles its site with probability ``1 - exp(-A_s)`` (then
    ``E_s += 1`` and it leaves) or moves to a neighbour ``s'`` with
    probability ``A_s' / sum A`` over the four neighbours; then ``B_s <- [(1 -
    eta) B_s + (eta/4) sum_{s'} B_s'] (1 - omega) + E_s`` (broken-windows
    spreading and repeat-victimisation decay), and new criminals appear at
    every site in Poisson numbers with mean ``gamma``. Randomness: Philox
    stream ``t`` of ``seed`` at step ``t`` supplies, in order, one uniform
    per criminal for the burglary decision, one per surviving criminal for
    the move, and one per site for the births.

    References
    ----------
    Short, M. B., D'Orsogna, M. R., Pasour, V. B., Tita, G. E., Brantingham,
    P. J., Bertozzi, A. L. and Chayes, L. B. (2008). A statistical model of
    criminal behavior. *Mathematical Models and Methods in Applied
    Sciences*, 18(supp01), 1249-1267.
    D'Orsogna, M. R. and Perc, M. (2015). Statistical physics of crime: a
    review. *Physics of Life Reviews*, 12, 1-21.

    Examples
    --------
    >>> r = short_crime_lattice(4, 3, gamma=0.5, seed=2)
    >>> len(r.B), len(r.burglaries)
    (16, 3)
    """
    N = n * n
    B = [0.0] * N if B0 is None else [float(v) for v in B0]
    crim = [] if criminals is None else [int(c) for c in criminals]
    counts, bseries = [], []

    def nb(s):
        i, j = divmod(s, n)
        return [((i - 1) % n) * n + j, ((i + 1) % n) * n + j, i * n + (j - 1) % n, i * n + (j + 1) % n]

    for t in range(steps):
        u = [float(v) for v in random_uniform(2 * len(crim) + N, seed=seed, stream=t)]
        k = 0
        E = [0.0] * N
        survivors = []
        for s in crim:
            if u[k] < 1 - math.exp(-(A0 + B[s])):
                E[s] += 1
            else:
                survivors.append(s)
            k += 1
        moved = []
        for s in survivors:
            nbs = nb(s)
            wts = [A0 + B[x] for x in nbs]
            tot = ssum(wts)
            r, acc, dest = u[k] * tot, 0.0, nbs[-1]
            for x, w in zip(nbs, wts):
                acc += w
                if r < acc:
                    dest = x
                    break
            moved.append(dest)
            k += 1
        k = 2 * len(crim)
        for s in range(N):
            moved.extend([s] * _poisson_inv(u[k + s], gamma))
        B = [((1 - eta) * B[s] + eta / 4 * ssum(B[x] for x in nb(s))) * (1 - omega) + E[s] for s in range(N)]
        crim = moved
        counts.append(ssum(E))
        bseries.append(ssum(B) / N)
    return RichResult(payload={"B": B, "criminals": crim, "burglaries": counts, "mean_B": bseries})


def short_crime_pde(
    B,
    rho,
    *,
    dx: float = 1.0,
    dt: float = 0.01,
    steps: int = 100,
    eta: float = 0.03,
    omega: float = 1 / 15,
    A0: float = 1 / 30,
    gamma: float = 0.002,
    D: float = 1.0,
    eps: float = 1.0,
    z: int = 4,
    suppress=None,
) -> RichResult:
    r"""Continuum Short et al. burglary model on a periodic grid (explicit finite differences).

    ``dB/dt = (eta D / z) lap B - omega B + eps D rho A`` and ``drho/dt = (D/z)
    div(grad rho - (2 rho / A) grad A) - rho A + gamma`` with ``A = A0 + B``
    (D'Orsogna and Perc 2015, eqs. 4-5). The advective flux uses face values
    ``(rho_i/A_i + rho_k/A_k)/2 (A_k - A_i)/dx`` so offender mass is
    conserved exactly by the transport term. ``suppress`` (a 0/1 grid) sets
    the crime rate ``rho A`` to zero at policed cells (hotspot suppression).
    The uniform steady state ``B* = eps D gamma / omega``, ``rho* = gamma / (A0 +
    B*)`` is returned for reference.

    Examples
    --------
    >>> r = short_crime_pde([[0.03] * 4] * 4, [[0.03] * 4] * 4, steps=10)
    >>> round(r.B_star, 6), round(r.rho_star, 6)
    (0.03, 0.031579)
    """
    Bg = [[float(v) for v in r] for r in B]
    Rg = [[float(v) for v in r] for r in rho]
    ny, nx = len(Bg), len(Bg[0])
    S = [[0.0] * nx for _ in range(ny)] if suppress is None else [[float(v) for v in r] for r in suppress]
    for _ in range(steps):
        A = [[A0 + b for b in r] for r in Bg]
        nB = [[0.0] * nx for _ in range(ny)]
        nR = [[0.0] * nx for _ in range(ny)]
        for i in range(ny):
            for j in range(nx):
                nbr = [((i - 1) % ny, j), ((i + 1) % ny, j), (i, (j - 1) % nx), (i, (j + 1) % nx)]
                lapB = (ssum(Bg[a][b] for a, b in nbr) - 4 * Bg[i][j]) / (dx * dx)
                lapR = (ssum(Rg[a][b] for a, b in nbr) - 4 * Rg[i][j]) / (dx * dx)
                adv = ssum((Rg[i][j] / A[i][j] + Rg[a][b] / A[a][b]) / 2 * (A[a][b] - A[i][j]) for a, b in nbr) / (
                    dx * dx
                )
                crime = 0.0 if S[i][j] else Rg[i][j] * A[i][j]
                nB[i][j] = Bg[i][j] + dt * (eta * D / z * lapB - omega * Bg[i][j] + eps * D * crime)
                nR[i][j] = Rg[i][j] + dt * (D / z * (lapR - 2 * adv) - crime + gamma)
        Bg, Rg = nB, nR
    bstar = eps * D * gamma / omega if omega > 0 else float("nan")
    return RichResult(payload={"B": Bg, "rho": Rg, "B_star": bstar, "rho_star": gamma / (A0 + bstar)})


def cheatsheet() -> str:
    return "hypercube_queue / square_root_law / short_crime_lattice / short_crime_pde -> police operations research."
