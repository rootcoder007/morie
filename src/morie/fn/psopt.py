# morie.fn -- function file (rootcoder007/morie)
"""
Particle swarm optimization for global optimization.

Population of particles move through search space with velocity updates.
"""

from . import _array_core as np

__all__ = ["psopt"]


def psopt(f, bounds, n_particles=30, generations=100, w=0.7, c1=1.5, c2=1.5, full_output=False, seed=None):
    """
    Particle swarm optimization (PSO) for global optimization.

    Particles move through search space with cognitive/social components.

    Parameters
    ----------
    f : callable
        Objective function f(x) to minimize.
    bounds : list of tuple
        [(x_min, x_max), ...] bounds for each dimension.
    n_particles : int, optional
        Number of particles (default 30).
    generations : int, optional
        Number of generations (default 100).
    w : float, optional
        Inertia weight (default 0.7).
    c1 : float, optional
        Cognitive parameter (default 1.5).
    c2 : float, optional
        Social parameter (default 1.5).
    full_output : bool, optional
        If True, return (x_min, info_dict).
    seed : int, optional
        Random seed.

    Returns
    -------
    x_min : ndarray
        Estimated minimizer.
    info_dict : dict, optional
        Dictionary with keys: 'generations', 'converged', 'final_value'.

    References
    ----------
    Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    In Proceedings of ICNN 1995 (Vol. 4, pp. 1942-1948). IEEE.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn import psopt
    >>> f = lambda x: (x[0] - 2)**2 + (x[1] - 3)**2
    >>> bounds = [(0, 5), (0, 5)]
    >>> x_min = psopt(f, bounds, n_particles=20, generations=50, seed=42)
    >>> np.allclose(x_min, [2, 3], atol=0.5)
    True
    """
    # A local stream: seeding the global generator would replace the
    # caller's random state for the rest of their session. RandomState
    # reproduces exactly what np.random.seed() + np.random.* produced.
    rs = np.random.RandomState(seed)

    n_vars = len(bounds)
    bounds = np.array(bounds)

    # Initialize
    x = rs.uniform(bounds[:, 0], bounds[:, 1], (n_particles, n_vars))
    v = rs.uniform(-1, 1, (n_particles, n_vars))
    fx = np.array([f(xi) for xi in x])

    # Best personal and global
    pbest = x.copy()
    fbest = fx.copy()
    best_idx = np.argmin(fbest)
    gbest = x[best_idx].copy()
    fgbest = fbest[best_idx]

    for gen in range(generations):
        for i in range(n_particles):
            r1 = rs.rand(n_vars)
            r2 = rs.rand(n_vars)
            v[i] = w * v[i] + c1 * r1 * (pbest[i] - x[i]) + c2 * r2 * (gbest - x[i])

            x[i] = x[i] + v[i]
            x[i] = np.clip(x[i], bounds[:, 0], bounds[:, 1])

            fx[i] = f(x[i])
            if fx[i] < fbest[i]:
                pbest[i] = x[i].copy()
                fbest[i] = fx[i]

            if fbest[i] < fgbest:
                gbest = pbest[i].copy()
                fgbest = fbest[i]

    if full_output:
        return gbest, {"generations": generations, "converged": False, "final_value": fgbest}
    return gbest


def cheatsheet() -> str:
    return "psopt(f, bounds, n_particles, generations, w, c1, c2, full_output, seed) -> Particle swarm optimization (PSO) for global optimization."
