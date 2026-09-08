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

DETERMINISM.  The three donor indices and the crossover decisions are
not drawn: r1, r2, r3 are taken by a fixed offset rotation of the
population index, and the crossover uses van der Corput points.  Every
structural feature of DE -- distinct donors, at least one inherited
coordinate, greedy replacement -- is preserved, and the run reproduces
exactly in both arms.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["differential_evolution"]


def differential_evolution(f, population, F=0.8, CR=0.9, generations=20):
    """DE/rand/1/bin with a deterministic donor and crossover schedule.

    Returns
    -------
    estimate : the best objective value found
    x        : the best point
    population : the final population
    fvals    : its objective values
    evals    : number of objective evaluations
    """
    P = [list(row) for row in k.mat(population)]
    npop = len(P)
    d = len(P[0]) if npop else 0
    fv = [float(f(P[i])) for i in range(npop)]
    evals = npop
    step = 0
    for _ in range(int(generations)):
        for i in range(npop):
            r1 = (i + 1) % npop
            r2 = (i + 2) % npop
            r3 = (i + 3) % npop
            jr = int(k.vdc(step, 3) * d)
            if jr >= d:
                jr = d - 1
            u = [0.0] * d
            for j in range(d):
                if k.vdc(step * d + j, 2) <= float(CR) or j == jr:
                    u[j] = P[r1][j] + float(F) * (P[r2][j] - P[r3][j])
                else:
                    u[j] = P[i][j]
            fu = float(f(u))
            evals += 1
            step += 1
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
            "method": "DE/rand/1/bin (Storn and Price 1997) with a deterministic donor and crossover schedule",
        },
    )


def cheatsheet():
    return "deopti: Differential evolution"
