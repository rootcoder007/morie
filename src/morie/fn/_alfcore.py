# morie.fn -- function file (rootcoder007/morie)
"""Shared layer algebra for the AlphaFold modules (Jumper et al. 2021).

Reference is the Supplementary Information of Jumper et al. (2021),
"Highly accurate protein structure prediction with AlphaFold", Nature
596:583-589, which specifies the network as numbered pseudocode
(Algorithms 1-32).

Everything here is deterministic and weight-free: every projection takes
its weight matrix from the caller, so the same weights give the same
answer in Python and in R.  Nothing is initialised randomly and no
trained value is baked in -- this is the layer algebra, not inference
with AlphaFold's published parameters.
"""

from __future__ import annotations

import math

__all__ = []


# ------------------------------------------------------------ elementwise
def sigm(x):
    """Logistic sigmoid, written to avoid overflow for large |x|."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def relu(x):
    return x if x > 0.0 else 0.0


def smax(v):
    """Softmax with max subtraction.

    The stabilisation is part of the contract: the R mirror subtracts the
    same maximum, otherwise the two arms disagree in the last bits.
    """
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


# ------------------------------------------------------------ vector ops
def vadd(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def vsub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def vscale(a, s):
    return [x * s for x in a]


def vdot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def vnorm2(a):
    """Squared L2 norm."""
    return sum(x * x for x in a)


# ------------------------------------------------------------ dense layers
def lin(v, W, b=None):
    """Dense projection of a vector; ``W`` is (n_out x n_in), rows first."""
    o = [sum(W[r][c] * v[c] for c in range(len(v))) for r in range(len(W))]
    if b is not None:
        o = [o[r] + b[r] for r in range(len(o))]
    return o


def lnorm(v, g=None, b=None, eps=1e-5):
    """Layer normalisation over the channel axis of one vector."""
    n = len(v)
    mu = sum(v) / n
    var = sum((x - mu) ** 2 for x in v) / n
    d = math.sqrt(var + eps)
    o = [(x - mu) / d for x in v]
    if g is not None:
        o = [o[i] * g[i] for i in range(n)]
    if b is not None:
        o = [o[i] + b[i] for i in range(n)]
    return o


# ------------------------------------------------------------ rigid frames
# A frame T is the pair (R, t) with R a 3x3 rotation as a list of rows and
# t a length-3 translation.  T o x = R x + t.
def rapply(T, x):
    R, t = T[0], T[1]
    return [R[r][0] * x[0] + R[r][1] * x[1] + R[r][2] * x[2] + t[r]
            for r in range(3)]


def rinv(T):
    """Inverse frame: (R', -R' t).  T^-1 o x = R'(x - t)."""
    R, t = T[0], T[1]
    Rt = [[R[0][r], R[1][r], R[2][r]] for r in range(3)]
    ti = [-(Rt[r][0] * t[0] + Rt[r][1] * t[1] + Rt[r][2] * t[2])
          for r in range(3)]
    return [Rt, ti]


def rinvapply(T, x):
    """T^-1 o x, without materialising the inverse frame."""
    R, t = T[0], T[1]
    d = [x[0] - t[0], x[1] - t[1], x[2] - t[2]]
    return [R[0][r] * d[0] + R[1][r] * d[1] + R[2][r] * d[2] for r in range(3)]


def rcompose(A, B):
    """Frame composition A o B, i.e. (A o B) o x = A o (B o x)."""
    RA, tA = A[0], A[1]
    RB, tB = B[0], B[1]
    R = [[sum(RA[i][k] * RB[k][j] for k in range(3)) for j in range(3)]
         for i in range(3)]
    return [R, rapply(A, tB)]


def quat2rot(b, c, d):
    """Non-unit quaternion (1, b, c, d) to a rotation matrix.

    Algorithm 23 lines 2-3.  The leading component is fixed to 1 before
    normalisation, which both guarantees a unit quaternion and makes the
    identity rotation the zero-input case.
    """
    n = math.sqrt(1.0 + b * b + c * c + d * d)
    a, b, c, d = 1.0 / n, b / n, c / n, d / n
    return [
        [a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
        [2 * b * c + 2 * a * d, a * a - b * b + c * c - d * d, 2 * c * d - 2 * a * b],
        [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a - b * b - c * c + d * d],
    ]


def ident():
    return [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0]]


# ------------------------------------------------------------ misc
def onehotnb(x, bins):
    """One-hot encoding with nearest bin -- Algorithm 5.

    Ties go to the lowest index, matching ``arg min`` in the pseudocode
    and ``which.min`` in R.
    """
    best, bi = abs(x - bins[0]), 0
    for k in range(1, len(bins)):
        dk = abs(x - bins[k])
        if dk < best:
            best, bi = dk, k
    p = [0.0] * len(bins)
    p[bi] = 1.0
    return p


def xent(y, p, eps=1e-12):
    """Cross entropy -sum(y log p) for one distribution."""
    return -sum(y[k] * math.log(p[k] if p[k] > eps else eps)
                for k in range(len(y)))
