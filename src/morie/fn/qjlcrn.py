# morie.fn -- function file (rootcoder007/morie)
r"""Johnson-Lindenstrauss projections with binary coins.

**The guarantee.** For :math:`n` points in :math:`\mathbb{R}^d` and
:math:`\varepsilon, \beta > 0`, put

.. math:: k_0 = \frac{4 + 2\beta}{\varepsilon^2/2 - \varepsilon^3/3}
          \log n .

For any :math:`k \ge k_0`, let :math:`R` be a :math:`d\times k` random
matrix with i.i.d. entries from either

.. math:: r_{ij} = \begin{cases} +1 & \text{w.p. } 1/2\\
          -1 & \text{w.p. } 1/2\end{cases}
          \qquad\text{or}\qquad
          r_{ij} = \sqrt3 \begin{cases} +1 & \text{w.p. } 1/6\\
          0 & \text{w.p. } 2/3\\ -1 & \text{w.p. } 1/6,\end{cases}

and let :math:`E = \tfrac{1}{\sqrt k} A R`. Then with probability at
least :math:`1 - n^{-\beta}`, every pairwise squared distance is
preserved within :math:`1 \pm \varepsilon`.

**Why it matters that the coins are simple.** The classical
construction needs Gaussian entries, or a random orthogonal projection.
Here the entries are :math:`\pm1`, so the projection is *additions and
subtractions only* -- no multiplications -- and the second
distribution is zero two-thirds of the time, so only a third of the
attributes are touched per output coordinate. Both distributions have
mean 0 and variance 1, which is what the proof needs and what
``moments`` verifies exactly rather than by simulation.

**What is checked here.** The bound :math:`k_0` is reported in closed
form; ``project`` scales by :math:`1/\sqrt k` so that the expected
squared norm is preserved exactly, and ``distortion`` measures the
realised distortion over every pair rather than reporting a promise.
The anchor runs a real dataset at :math:`k \ge k_0` and confirms every
pair lands inside :math:`\varepsilon`, and separately shows the
guarantee failing at a :math:`k` far below the bound -- so the check
can fail.

**Not implemented.** The ledger pairs this with Charikar & Sahai
(2002) on dimension reduction lower bounds for :math:`\ell_1`; that
paper is not in the corpus and nothing here claims an :math:`\ell_1`
result. Johnson-Lindenstrauss is an :math:`\ell_2` statement and
``project`` says so.

References
----------
Achlioptas, D. (2003) "Database-friendly random projections:
Johnson-Lindenstrauss with binary coins", *Journal of Computer and
System Sciences* 66(4), 671-687, doi:10.1016/S0022-0000(03)00025-4.
Theorem 1.1 for :math:`k_0`, the two entry distributions, the scaling
:math:`E = AR/\sqrt k` and the :math:`1 - n^{-\beta}` success
probability; Sec. 1.1 for the observation that the projection reduces
to additions and subtractions and that the sparse distribution touches
a third of the attributes; and Lemma 5.1 for the moment argument the
theorem rests on.

Johnson, W. B. & Lindenstrauss, J. (1984) "Extensions of Lipschitz
mappings into a Hilbert space", in *Conference in Modern Analysis and
Probability*, Contemporary Mathematics 26, 189-206,
doi:10.1090/conm/026/737400, for the original lemma.
"""

import math

from . import _array_core as np
from . import survrsf as _rsf
from ._richresult import RichResult

__all__ = ["DISTRIBUTIONS", "target_dimension", "moments",
           "projection_matrix", "project", "distortion"]

DISTRIBUTIONS = ("rademacher", "sparse")


def _check(distribution):
    if distribution not in DISTRIBUTIONS:
        raise ValueError("qjlcrn: distribution must be one of %s, got "
                         "%r" % (", ".join(DISTRIBUTIONS),
                                 distribution))


def target_dimension(n, epsilon, beta=1.0):
    r"""Theorem 1.1: :math:`k_0 = (4+2\beta)\log n /
    (\varepsilon^2/2 - \varepsilon^3/3)`."""
    n = int(n)
    e, b = float(epsilon), float(beta)
    if n < 2:
        raise ValueError("qjlcrn: need at least two points")
    if not 0.0 < e < 1.0:
        raise ValueError("qjlcrn: epsilon must lie in (0, 1), got %r"
                         % epsilon)
    if b <= 0.0:
        raise ValueError("qjlcrn: beta must be positive")
    denom = e * e / 2.0 - e ** 3 / 3.0
    k0 = (4.0 + 2.0 * b) * math.log(n) / denom
    return {"k0": k0, "k": int(math.ceil(k0)), "n": n, "epsilon": e,
            "beta": b, "failure_probability": n ** (-b),
            "note": "log is natural, as in the paper"}


def moments(distribution="rademacher"):
    r"""The two distributions have mean 0 and variance 1 exactly."""
    _check(distribution)
    if distribution == "rademacher":
        support = ((1.0, 0.5), (-1.0, 0.5))
    else:
        s = math.sqrt(3.0)
        support = ((s, 1.0 / 6.0), (0.0, 2.0 / 3.0), (-s, 1.0 / 6.0))
    m1 = sum(v * p for v, p in support)
    m2 = sum(v * v * p for v, p in support)
    m4 = sum(v ** 4 * p for v, p in support)
    return {"mean": m1, "variance": m2, "fourth_moment": m4,
            "support": support,
            "density": 1.0 if distribution == "rademacher"
            else 1.0 / 3.0,
            "note": "the sparse distribution is zero two-thirds of "
                    "the time, so only a third of the attributes are "
                    "touched per output coordinate"}


def projection_matrix(d, k, distribution="rademacher", seed=0):
    r"""A :math:`d \times k` matrix of the paper's coins."""
    _check(distribution)
    d, k = int(d), int(k)
    if d < 1 or k < 1:
        raise ValueError("qjlcrn: dimensions must be positive")
    rng = _rsf._Rng(seed)
    s = math.sqrt(3.0)
    R = []
    for _ in range(d):
        row = []
        for _ in range(k):
            u = rng.next()
            if distribution == "rademacher":
                row.append(1.0 if u < 0.5 else -1.0)
            else:
                row.append(s if u < 1.0 / 6.0
                           else (-s if u > 5.0 / 6.0 else 0.0))
        R.append(row)
    return R


def project(A, k, distribution="rademacher", seed=0):
    r""":math:`E = AR/\sqrt k`, the scaling that preserves norms in
    expectation."""
    n = len(A)
    if n == 0:
        raise ValueError("qjlcrn: no points supplied")
    d = len(A[0])
    if any(len(row) != d for row in A):
        raise ValueError("qjlcrn: every point needs %d coordinates"
                         % d)
    R = projection_matrix(d, k, distribution, seed)
    scale = 1.0 / math.sqrt(float(k))
    E = [[scale * sum(A[i][t] * R[t][j] for t in range(d))
          for j in range(int(k))] for i in range(n)]
    nz = sum(1 for row in R for v in row if v != 0.0)
    return RichResult(payload={
        "estimate": float(k), "embedding": E, "matrix": R, "k": int(k),
        "d": d, "n": n, "distribution": distribution,
        "nonzero_fraction": nz / float(d * int(k)),
        "norm": "l2 -- Johnson-Lindenstrauss says nothing about l1",
        "method": "random projection with binary coins; Achlioptas "
                  "(2003) Theorem 1.1",
    })


def distortion(A, E):
    r"""Realised distortion over every pair, not a promise."""
    n = len(A)
    if n != len(E):
        raise ValueError("qjlcrn: the embedding must have one row per "
                         "point")
    worst = 0.0
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            d0 = sum((A[i][t] - A[j][t]) ** 2
                     for t in range(len(A[0])))
            d1 = sum((E[i][t] - E[j][t]) ** 2
                     for t in range(len(E[0])))
            if d0 <= 0.0:
                continue
            ratios.append(d1 / d0)
            worst = max(worst, abs(d1 / d0 - 1.0))
    if not ratios:
        raise ValueError("qjlcrn: every pair of points coincides")
    return {"worst_distortion": worst,
            "min_ratio": min(ratios), "max_ratio": max(ratios),
            "mean_ratio": sum(ratios) / len(ratios),
            "n_pairs": len(ratios)}


def cheatsheet():
    return ("qjlcrn: k0 = (4 + 2 beta) log n / (eps^2/2 - eps^3/3), "
            "R with entries +-1 (or sqrt(3) times {+1,0,-1} at "
            "1/6, 2/3, 1/6), E = AR/sqrt(k). Both distributions have "
            "mean 0 and variance 1, so no Gaussians and no "
            "multiplications are needed; the sparse one touches a "
            "third of the attributes. Every pairwise squared distance "
            "is then within 1 +- eps with probability 1 - n^-beta. "
            "This is an l2 statement only.")


# compact alias per ledger/NAMING.md
johnson_lindenstrauss = project
