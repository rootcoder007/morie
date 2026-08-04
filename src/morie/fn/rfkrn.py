# morie.fn -- slice s04 (rootcoder007/morie)
"""Random Fourier features (RFF) kernel approximation.

NOT IN THE BOOK.  Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, was searched in full -- all seventeen page-range
volumes and the index, [Pages 683-691].  Chapter 8, volume [Pages
251-336], is the kernel chapter and builds every kernel it uses as an
explicit n-by-n matrix; it never approximates one by a feature map, and
"random Fourier" and "Bochner" do not occur anywhere.

The method is therefore taken from the originating primary source,
Rahimi, A. and Recht, B. (2007), Random features for large-scale kernel
machines, *Advances in Neural Information Processing Systems* 20 (NIPS
2007), pp. 1177-1184.  It is a NIPS proceedings paper and carries no DOI.

CITATION CARE.  That paper gives exactly ONE feature map, its equation
for z(x) in Algorithm 1:

    z(x) = sqrt(2/D) [ cos(w_1'x + b_1), ..., cos(w_D'x + b_D) ]',

with w_1, ..., w_D drawn from p(w), the Fourier transform of the kernel,
and b_1, ..., b_D drawn from Uniform(0, 2*pi).  The 2D-dimensional
sin/cos map -- [sin(w'x), cos(w'x)] stacked without a phase offset --
that is often attributed to this paper is NOT in it, and is not
implemented here.

The spectral density is the paper's own Gaussian entry: for
k(x - y) = exp(-||x - y||^2 / 2) the density p(w) is N(0, I).  The
bandwidth convention matters and is stated rather than assumed.  This
function takes the kernel in the gamma parameterisation
k(x, y) = exp(-gamma ||x - y||^2), which is the convention
kernel="rbf" uses throughout this package; matching exponents,
exp(-gamma||d||^2) = exp(-||d||^2/2) requires the spectral draws to be
scaled by sqrt(2*gamma), so w ~ N(0, 2*gamma*I).  At gamma = 1/2 that
reduces to the paper's N(0, I).

DETERMINISM.  The paper draws w and b at random.  Both are replaced here
by a Halton sequence -- van der Corput in a DIFFERENT PRIME BASE for each
coordinate, all indexed by the same j -- mapped through the inverse
normal for w and scaled to (0, 2*pi) for b.  That is deterministic,
identical in both arms, and being low-discrepancy it converges faster in
D than the paper's own Monte Carlo draw rather than slower.

The separate bases are load-bearing, not cosmetic.  Writing
cos(A)cos(B) = [cos(A - B) + cos(A + B)]/2 gives

    z(x)'z(y) = (1/D) sum_j cos(w_j'(x - y))
              + (1/D) sum_j cos(w_j'(x + y) + 2 b_j),

and the estimator is only unbiased because the SECOND sum vanishes, which
needs b independent of w.  A first attempt strided a single van der
Corput stream across the coordinates, so b and one column of W came from
the same base-3 sequence at interleaved indices.  They were correlated,
the second sum did not vanish, and the mean absolute error against the
exact kernel GREW with D -- 0.051 at D = 64, 0.066 at D = 512, 0.070 at
D = 4096 -- converging to the wrong target.  Both arms would have agreed
on it to 1e-16.  One base per coordinate fixes it and the error now falls
with D as it must.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["random_fourier_features"]


def _primes(k):
    """The first k primes, for the Halton bases."""
    out = []
    c = 2
    while len(out) < k:
        j = 2
        ok = True
        while j * j <= c:
            if c % j == 0:
                ok = False
                break
            j += 1
        if ok:
            out.append(c)
        c += 1
    return out


def random_fourier_features(X, D=256, kernel="rbf", gamma=0.5):
    """z(x) = sqrt(2/D) cos(W'x + b), the Rahimi and Recht (2007) map.

    Parameters
    ----------
    X : array-like
        n-by-p matrix of inputs.
    D : int
        Number of random features.
    kernel : str
        Only "rbf" is offered, the Gaussian entry of the paper's table.
    gamma : float
        The kernel is exp(-gamma ||x - y||^2); gamma = 1/2 is the paper's
        own unit-variance Gaussian.

    Returns
    -------
    estimate : the mean absolute error of Z Z' against the exact kernel
    Z        : the n-by-D feature matrix
    K_approx : Z Z', the approximated kernel
    K_exact  : the exact Gaussian kernel, for comparison
    W        : the p-by-D spectral matrix actually used
    b        : the D phase offsets actually used
    """
    XX = core.mat(X)
    n = len(XX)
    if n == 0:
        raise ValueError("random_fourier_features: X is empty")
    p = len(XX[0])
    if p == 0:
        raise ValueError("random_fourier_features: X has no columns")
    for r in XX:
        if len(r) != p:
            raise ValueError("random_fourier_features: X rows have unequal lengths")
    d = int(D)
    if d < 1:
        raise ValueError("random_fourier_features: D must be at least 1")
    if kernel != "rbf":
        raise ValueError("random_fourier_features: only the rbf kernel of the paper's "
                         "Gaussian entry is offered")
    g = float(gamma)
    if g <= 0.0:
        raise ValueError("random_fourier_features: gamma must be positive")
    # w ~ N(0, 2*gamma*I) so that exp(-gamma||d||^2) is the target kernel.
    # One prime base per coordinate, plus one more for the phase: a Halton
    # sequence in p + 1 dimensions.  See the docstring on why sharing a base
    # between W and b biases the estimator.
    scale = math.sqrt(2.0 * g)
    pr = _primes(p + 1)
    W = [[0.0] * d for _ in range(p)]
    for j in range(d):
        for a in range(p):
            W[a][j] = scale * core.qnorm(core.vdc(j + 1, pr[a]))
    b = [2.0 * math.pi * core.vdc(k + 1, pr[p]) for k in range(d)]
    c = math.sqrt(2.0 / d)
    Z = []
    for i in range(n):
        row = []
        for j in range(d):
            s = b[j]
            for a in range(p):
                s += XX[i][a] * W[a][j]
            row.append(c * math.cos(s))
        Z.append(row)
    Ka = [[0.0] * n for _ in range(n)]
    Ke = [[0.0] * n for _ in range(n)]
    err = 0.0
    for i in range(n):
        for k in range(n):
            s = 0.0
            for j in range(d):
                s += Z[i][j] * Z[k][j]
            Ka[i][k] = s
            q = 0.0
            for a in range(p):
                dd = XX[i][a] - XX[k][a]
                q += dd * dd
            Ke[i][k] = math.exp(-g * q)
            err += abs(s - Ke[i][k])
    return RichResult(
        title="Random Fourier features",
        summary_lines=[("rows", n), ("dim", p), ("features", d)],
        payload={
            "estimate": err / (n * n),
            "Z": Z,
            "K_approx": Ka,
            "K_exact": Ke,
            "W": W,
            "b": b,
            "n": n,
            "method": "z(x) = sqrt(2/D) cos(W'x + b), w ~ N(0, 2*gamma I), b ~ U(0, 2pi); "
                      "Rahimi and Recht (2007) NIPS 20; not in the book",
        },
    )


def cheatsheet():
    return "rfkrn: Random Fourier features (RFF) kernel approximation"
