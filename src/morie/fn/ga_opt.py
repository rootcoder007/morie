# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Genetic algorithm with elitist truncation selection.

Holland (1975), *Adaptation in Natural and Artificial Systems*,
University of Michigan Press.  One generation is selection, crossover
and mutation:

    keep the best ceil(m/2) individuals unchanged (elitism),
    each child takes a prefix from one parent and a suffix from
    another (one-point crossover),
    then a mutation is added to the child.

The randomness is supplied by the van der Corput low-discrepancy
sequence rather than a pseudo-random generator, so the run is exactly
reproducible and both language arms visit the same individuals.
Elitism makes the best fitness monotone, which the tests assert.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["genetic_algorithm"]


def genetic_algorithm(f, population, generations=20, mutation=0.1):
    """Minimise f over a population of candidate vectors."""
    P = core.mat(population)
    m = len(P)
    if m < 2:
        raise ValueError("genetic_algorithm: population needs at least two individuals")
    d = len(P[0])
    if d == 0:
        raise ValueError("genetic_algorithm: individuals are empty")
    if not callable(f):
        raise ValueError("genetic_algorithm: f must be callable")
    ng = int(generations)
    if ng < 1:
        raise ValueError("genetic_algorithm: generations must be at least 1")
    mu = float(mutation)
    h = (m + 1) // 2
    counter = 0
    best_path = []
    for _ in range(ng):
        fit = [float(f(P[i])) for i in range(m)]
        order = sorted(range(m), key=lambda i: (fit[i], i))
        keep = [P[order[i]] for i in range(h)]
        best_path.append(fit[order[0]])
        kids = []
        for i in range(m - h):
            a = keep[i % h]
            b = keep[(i + 1) % h]
            cp = (i % (d - 1)) + 1 if d > 1 else 0
            child = [a[j] if j < cp else b[j] for j in range(d)]
            for j in range(d):
                child[j] += mu * (2.0 * core.vdc(counter) - 1.0)
                counter += 1
            kids.append(child)
        P = [row[:] for row in keep] + kids
    fit = [float(f(P[i])) for i in range(m)]
    order = sorted(range(m), key=lambda i: (fit[i], i))
    best_path.append(fit[order[0]])
    return RichResult(
        title="Genetic algorithm",
        summary_lines=[("population", m), ("generations", ng)],
        payload={
            "estimate": fit[order[0]],
            "best": P[order[0]],
            "best_fitness": fit[order[0]],
            "best_path": best_path,
            "generations": ng,
            "n": m,
            "method": "elitist truncation selection, one-point crossover, van der Corput mutation; Holland (1975)",
        },
    )


def cheatsheet():
    return "ga_opt: genetic algorithm"
