# morie.fn -- function file (rootcoder007/morie)
"""Shared helpers for the big1 quantization modules.

Internal only.  A deterministic rotation matrix and a Lloyd-Max scalar
codebook, both needed by more than one module in this batch, kept here
rather than duplicated.  Mirrors ``aaa_b1_turbo.R`` on the R side.
"""

import math

from . import _tail1core as C

__all__ = []


def rotation(d, seed=1):
    """A deterministic orthogonal matrix, via Gram-Schmidt on LCG normals.

    The paper asks for "a random rotation matrix"; a genuinely random
    one cannot be compared across languages, so the randomness is
    pinned to the shared Lehmer stream and the factor is sign-fixed
    (positive diagonal of R) which makes the QR factor unique.
    """
    d = int(d)
    if d < 1:
        raise ValueError("the dimension must be at least 1")
    g = C.Lcg(seed)
    A = [[g.norm() for _ in range(d)] for _ in range(d)]
    Q = [[0.0] * d for _ in range(d)]
    for j in range(d):
        v = [A[i][j] for i in range(d)]
        for k in range(j):
            p = sum(Q[i][k] * A[i][j] for i in range(d))
            for i in range(d):
                v[i] -= p * Q[i][k]
        nrm = math.sqrt(sum(x * x for x in v))
        if nrm < 1e-300:
            raise ValueError("the random matrix was rank deficient")
        # Sign-fix on the pivot so the factorisation is unique.
        if v[j] < 0.0:
            nrm = -nrm
        for i in range(d):
            Q[i][j] = v[i] / nrm
    return Q


def codebook(b, iters=200, grid=2001, lo=-6.0, hi=6.0):
    """Lloyd-Max centroids for a standard normal source, 2^b levels.

    The paper specifies the codebook only as "centroids that minimize
    MSE cost", i.e. the Lloyd-Max quantizer for the coordinate
    distribution.  After a random rotation a unit vector's coordinates
    are asymptotically normal, so the source is taken to be standard
    normal and the result rescaled by the caller.

    The Lloyd iteration runs a FIXED number of steps on a FIXED grid,
    never to a tolerance, so both language arms land on identical
    centroids.
    """
    b = int(b)
    if b < 1:
        raise ValueError("the bit width must be at least 1")
    K = 2 ** b
    n = int(grid)
    h = (hi - lo) / (n - 1)
    x = [lo + i * h for i in range(n)]
    w = [math.exp(-0.5 * v * v) for v in x]
    # Initial centroids at equally spaced quantiles of the source.
    c = [C.qnorm((k + 0.5) / K) for k in range(K)]
    for _ in range(int(iters)):
        num = [0.0] * K
        den = [0.0] * K
        for i in range(n):
            best = 0
            bd = abs(x[i] - c[0])
            for k in range(1, K):
                dk = abs(x[i] - c[k])
                if dk < bd:
                    bd = dk
                    best = k
            num[best] += w[i] * x[i]
            den[best] += w[i]
        for k in range(K):
            if den[k] > 0.0:
                c[k] = num[k] / den[k]
    return c


def quantize(y, c):
    """Nearest-centroid index of every coordinate; ties to the lower index."""
    K = len(c)
    out = []
    for v in y:
        best = 0
        bd = abs(v - c[0])
        for k in range(1, K):
            dk = abs(v - c[k])
            if dk < bd:
                bd = dk
                best = k
        out.append(best)
    return out
