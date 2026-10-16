# morie.fn -- slice s03 (rootcoder007/morie)
"""Differential evolution.

Source consulted: Storn, R. and Price, K. (1997).  Differential
evolution -- a simple and efficient heuristic for global optimization
over continuous spaces.  *Journal of Global Optimization* 11(4),
341-359.  The DE/rand/1/bin scheme they define is

    v_i = x_(r1) + F ( x_(r2) - x_(r3) ),   r1, r2, r3 distinct, != i
    u_(i,j) = v_(i,j) if rand_j <= CR or j = j_rand, else x_(i,j)
    x_i <- u_i  iff  f(u_i) <= f(x_i)

The paper is paywalled; the mutation, the binomial crossover including
the forced index j_rand, and the greedy selection are quoted in their
standard published form.

RANDOMNESS.  Donors and crossover draws come from the morie Philox
stream (seed, stream 0), consumed in a fixed order: for each trial
vector, three uniforms pick r1, r2, r3 without replacement from the
population minus i, one picks j_rand, then one per coordinate decides
crossover.  Both language arms read the same stream, so runs agree
exactly across arms.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k
from ._richresult import RichResult
from ._rng import random_uniform

__all__ = ["differential_evolution"]


def differential_evolution(f, population, F=0.8, CR=0.9, generations=20, seed=0):
    """DE/rand/1/bin (Storn and Price 1997) with Philox-drawn donors and crossover.

    Parameters
    ----------
    f : callable
        Objective to minimise.
    population : list of lists
        Starting population, one row per individual (at least 4 rows).
    F : float
        Differential weight.
    CR : float
        Crossover probability.
    generations : int
    seed : int
        Philox seed.

    Returns
    -------
    estimate : the best objective value found
    x        : the best point
    population : the final population
    fvals    : its objective values
    evaluations    : number of objective evaluations

    Examples
    --------
    >>> pop = [[2.0, -1.0], [-1.5, 0.5], [0.5, 1.5], [1.0, 1.0], [-2.0, -2.0], [0.2, -1.3]]
    >>> differential_evolution(lambda v: v[0] ** 2 + v[1] ** 2, pop, generations=100)["estimate"] < 1e-6
    True
    """
    P = [list(row) for row in k.mat(population)]
    npop = len(P)
    d = len(P[0]) if npop else 0
    if npop < 4:
        raise ValueError("DE/rand/1 needs at least 4 individuals")
    fv = [float(f(P[i])) for i in range(npop)]
    evals = npop
    per = 4 + d
    U = random_uniform(int(generations) * npop * per, seed=seed, stream=0)
    pos = 0
    for _ in range(int(generations)):
        for i in range(npop):
            pool = [q for q in range(npop) if q != i]
            r = []
            for t in range(3):
                r.append(pool.pop(min(int(U[pos + t] * len(pool)), len(pool) - 1)))
            r1, r2, r3 = r
            jr = min(int(U[pos + 3] * d), d - 1)
            u = [0.0] * d
            for j in range(d):
                if U[pos + 4 + j] <= float(CR) or j == jr:
                    u[j] = P[r1][j] + float(F) * (P[r2][j] - P[r3][j])
                else:
                    u[j] = P[i][j]
            pos += per
            fu = float(f(u))
            evals += 1
            if fu <= fv[i]:
                P[i] = u
                fv[i] = fu
    best = 0
    for i in range(1, npop):
        if fv[i] < fv[best]:
            best = i
    return RichResult(
        title="Differential evolution",
        summary_lines=[("best f", fv[best] if npop else float("nan"))],
        payload={
            "estimate": fv[best] if npop else float("nan"),
            "x": P[best] if npop else [],
            "population": P,
            "fvals": fv,
            "evals": evals,
            "method": "DE/rand/1/bin (Storn and Price 1997), Philox donors and crossover",
        },
    )


def cheatsheet():
    return "deopti: Differential evolution"
