# morie.fn -- function file (rootcoder007/morie)
"""Particle swarm optimisation on a deterministic low-discrepancy stream."""

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["particle_swarm"]

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def particle_swarm(f, bounds, n_particles=20, w=0.7, c1=1.5, c2=1.5,
                   maxiter=200):
    """Swarm search whose "randomness" is a Halton stream, not a PRNG.

    ``morie.fn.pswrm.particle_swarm`` is the seeded stochastic version of
    this method.  It cannot be checked across languages: its numbers come
    from Python's native generator stream, and the R tree's Philox
    produces a different sequence from the same seed, so a numeric parity
    check between the two arms is impossible by construction.  This
    module is the deterministic sibling -- the same algorithm, driven by
    a van der Corput sequence with a distinct prime base per coordinate,
    so both language arms visit the SAME points in the SAME order.

    Formula: ``v <- w v + c1 r1 (p_best - x) + c2 r2 (g_best - x)``,
    ``x <- clip(x + v, lo, hi)``, with ``r1``, ``r2`` drawn from the
    low-discrepancy stream instead of a uniform generator.

    Parameters
    ----------
    f : callable
        Objective ``f(x) -> float``, minimised.
    bounds : sequence of (lo, hi)
        Search box, one pair per dimension; at most 16 dimensions, since
        that is how many prime bases are carried.
    n_particles : int, default 20
        Swarm size, at least 1.
    w, c1, c2 : float
        Inertia, cognitive and social coefficients, non-negative.
    maxiter : int, default 200
        Iterations, non-negative.

    Returns
    -------
    RichResult
        ``estimate`` (the best objective value), ``value``, ``x`` (the
        best point), ``n_eval``, ``n_particles``, ``maxiter``, ``d``.

    References
    ----------
    Kennedy, J. & Eberhart, R. (1995).  Particle swarm optimization.
    Proceedings of ICNN'95 -- International Conference on Neural
    Networks, volume 4, pages 1942-1948.
    doi:10.1109/ICNN.1995.488968.
    """
    if not callable(f):
        raise ValueError("particle_swarm: f must be callable")
    bnd = [(float(a), float(b)) for a, b in bounds]
    d = len(bnd)
    if d == 0:
        raise ValueError("particle_swarm: bounds is empty")
    if d > len(_PRIMES):
        raise ValueError("particle_swarm: at most 16 dimensions are supported")
    for lo, hi in bnd:
        if not (hi > lo):
            raise ValueError("particle_swarm: every bound needs hi > lo")
    n_particles = int(n_particles)
    maxiter = int(maxiter)
    if n_particles < 1:
        raise ValueError("particle_swarm: n_particles must be at least 1")
    if maxiter < 0:
        raise ValueError("particle_swarm: maxiter must be non-negative")
    w = float(w); c1 = float(c1); c2 = float(c2)
    if w < 0.0 or c1 < 0.0 or c2 < 0.0:
        raise ValueError("particle_swarm: coefficients must be non-negative")

    pos = []
    for i in range(n_particles):
        pos.append([bnd[j][0] + (bnd[j][1] - bnd[j][0]) * core.vdc(i + 1, _PRIMES[j])
                    for j in range(d)])
    vel = [[0.0] * d for _ in range(n_particles)]
    pbest = [list(p) for p in pos]
    pval = [float(f(p)) for p in pos]
    n_eval = n_particles
    gi = 0
    for i in range(n_particles):
        if pval[i] < pval[gi]:
            gi = i
    gbest = list(pbest[gi])
    gval = pval[gi]
    k = 1
    for _ in range(maxiter):
        for i in range(n_particles):
            for j in range(d):
                r1 = core.vdc(k, 2)
                r2 = core.vdc(k, 3)
                k += 1
                vel[i][j] = (w * vel[i][j]
                             + c1 * r1 * (pbest[i][j] - pos[i][j])
                             + c2 * r2 * (gbest[j] - pos[i][j]))
                v = pos[i][j] + vel[i][j]
                if v < bnd[j][0]:
                    v = bnd[j][0]
                elif v > bnd[j][1]:
                    v = bnd[j][1]
                pos[i][j] = v
            val = float(f(pos[i]))
            n_eval += 1
            if val < pval[i]:
                pval[i] = val
                pbest[i] = list(pos[i])
                if val < gval:
                    gval = val
                    gbest = list(pos[i])
    return RichResult(payload={
        "estimate": gval, "value": gval, "x": gbest, "n_eval": n_eval,
        "n_particles": n_particles, "maxiter": maxiter, "d": d,
        "method": "Particle swarm on a van der Corput stream"})


def cheatsheet():
    return "pso_op: Particle swarm optimisation (deterministic stream)"

# public names resolved by fn/_lazy_map.json
particleswarm = particle_swarm
