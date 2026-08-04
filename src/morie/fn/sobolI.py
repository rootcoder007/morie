# morie.fn -- slice s03 (rootcoder007/morie)
"""Sobol global sensitivity indices.

Sources consulted: Sobol, I. M. (1993).  Sensitivity estimates for
nonlinear mathematical models.  *Mathematical Modelling and
Computational Experiments* 1(4), 407-414, for the decomposition and the
indices S_i = V_i / V; and Saltelli, A. et al. (2010).  Variance based
sensitivity analysis of model output.  *Computer Physics Communications*
181(2), 259-270, for the estimators used here,

    V_i    = (1/N) sum_j f(B)_j ( f(A_B^(i))_j - f(A)_j )
    V_T,i  = (1/(2N)) sum_j ( f(A)_j - f(A_B^(i))_j )^2

with A and B two independent sample matrices and A_B^(i) the matrix A
with its i-th column replaced by B's.  Neither source was retrievable
here as a full text; both estimators are quoted in their standard
published form.

DETERMINISM.  A and B are not pseudo-random: they are the Sobol
sequence's own low-discrepancy points, generated here as van der Corput
sequences in distinct prime bases, which is the quasi-Monte Carlo design
the method is built for and is identical in both arms.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["sobol_indices"]

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def sobol_indices(model, input_dist=None, N=64, d=None):
    """First-order and total Sobol indices by the Saltelli estimators.

    Parameters
    ----------
    model : callable
        f(x) for a length-d vector x on the unit cube.
    input_dist : list of callable, optional
        Per-dimension inverse CDFs applied to the unit-cube points.
    N : int
        Base sample size.
    d : int, optional
        Input dimension; inferred from ``input_dist`` when absent.

    Returns
    -------
    estimate : S_1
    S        : first-order indices
    ST       : total indices
    V        : total output variance
    """
    dd = int(d) if d is not None else (len(input_dist) if input_dist else 2)
    n = int(N)
    # A and B must be INDEPENDENT samples.  Continuing one low-discrepancy
    # sequence gives points that are not: with base 2 and n a power of two,
    # vdc(j + n) and vdc(j) share their leading bits, the estimator's cross
    # terms stop cancelling, and S_i comes out badly wrong.  A and B
    # therefore use disjoint prime bases -- the standard split of a
    # 2d-dimensional quasi-Monte Carlo design into its two halves.
    A = [[k.vdc(j, _PRIMES[a]) for a in range(dd)] for j in range(n)]
    B = [[k.vdc(j, _PRIMES[dd + a]) for a in range(dd)] for j in range(n)]

    def tf(row):
        if input_dist is None:
            return list(row)
        return [input_dist[a](row[a]) for a in range(dd)]

    fA = [float(model(tf(A[j]))) for j in range(n)]
    fB = [float(model(tf(B[j]))) for j in range(n)]
    V = k.variance(fA + fB, 1)
    S = []
    ST = []
    for i in range(dd):
        AB = [[B[j][a] if a == i else A[j][a] for a in range(dd)]
              for j in range(n)]
        fAB = [float(model(tf(AB[j]))) for j in range(n)]
        vi = 0.0
        vti = 0.0
        for j in range(n):
            vi += fB[j] * (fAB[j] - fA[j]) / n
            vti += (fA[j] - fAB[j]) ** 2 / (2.0 * n)
        S.append(vi / V if V > 0.0 else float("nan"))
        ST.append(vti / V if V > 0.0 else float("nan"))
    return RichResult(
        title="Sobol sensitivity indices",
        summary_lines=[("dimensions", dd), ("N", n)],
        payload={
            "estimate": S[0] if S else float("nan"),
            "S": S,
            "ST": ST,
            "V": V,
            "n": n,
            "method": "Saltelli et al. (2010) estimators of the Sobol (1993) indices, on a quasi-Monte Carlo design",
        },
    )


def cheatsheet():
    return "sobolI: Sobol global sensitivity indices"


sobolindices = sobol_indices
