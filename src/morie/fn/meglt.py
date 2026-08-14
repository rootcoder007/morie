# morie.fn -- function file (rootcoder007/morie)
r"""Exact matrix completion by nuclear norm minimisation.

Most low-rank matrices can be recovered *perfectly* from what looks
like far too few entries. If :math:`m` entries of an :math:`n\times n`
matrix of rank :math:`r` are sampled uniformly and

.. math:: m \ge C\, n^{1.2} r \log n,

then with very high probability the matrix is the unique solution of a
convex program. Replacing the exponent 1.2 by 1.25 makes the statement
hold for **all** ranks rather than only moderate ones -- the weaker
exponent is the price of covering the high-rank case.

**Rank minimisation is NP-hard, so minimise its convex surrogate.**
The rank is the number of non-zero singular values; the nuclear norm

.. math:: \|X\|_* = \sum_{k} \sigma_k(X)

is their sum, and it is to the rank what the :math:`\ell_1` norm is to
the count of non-zeros in compressed sensing. The program is

.. math:: \min \|X\|_* \quad \text{subject to}\quad
          X_{ij} = M_{ij},\ (i,j) \in \Omega .

**Incoherence is not fine print.** A matrix whose singular vectors are
concentrated on a few coordinates -- :math:`e_1e_1^\top`, say -- is
low-rank and unrecoverable: almost every sampled entry is zero and
tells you nothing. The guarantee therefore requires the singular
vectors to be *spread*, and ``coherence`` computes exactly the
quantity that fails on such a matrix, which the anchor checks on both
a spread and a spiked example.

**Singular value thresholding.** Solving the program directly needs a
semidefinite solver; the standard first-order route iterates soft
thresholding of the singular values with a projection onto the
observed entries, which is what ``svt`` does.

References
----------
Candes, E. J. & Recht, B. (2009) "Exact Matrix Completion via Convex
Optimization", *Foundations of Computational Mathematics* 9(6),
717-772, doi:10.1007/s10208-009-9045-5, arXiv:0805.4471. The abstract
and Sec. 1 (the sampling bound m >= C n^{1.2} r log n, that the 1.25
exponent covers all ranks, the nuclear norm of eq. (1.4) as the sum of
singular values and its use in place of the rank, the connection to
compressed sensing, and the incoherence conditions -- with the
motivating example of a matrix whose singular vectors are extremely
sparse, for which sampling reveals nothing).

Cai, J.-F., Candes, E. J. & Shen, Z. (2010) "A Singular Value
Thresholding Algorithm for Matrix Completion", *SIAM Journal on
Optimization* 20(4), 1956-1982, doi:10.1137/080738970,
arXiv:0810.3286. The iterative algorithm implemented here.

Fazel, M. (2002) *Matrix Rank Minimization with Applications*, PhD
thesis, Stanford University. The nuclear norm as the convex envelope
of the rank.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["nuclear_norm", "coherence", "sample_bound", "svt",
           "relative_error"]

_EPS = 1e-12


def _svd(A):
    return np.linalg.svd(A, full_matrices=False)


def nuclear_norm(A):
    r""":math:`\|A\|_* = \sum_k \sigma_k(A)`."""
    _, s, _ = _svd([[float(v) for v in r] for r in k.mat(A)])
    return float(sum(s))


def coherence(A, rank=None):
    r"""How spread the singular vectors are.

    :math:`\mu = \frac{n}{r}\max_i \|P_U e_i\|^2`. A matrix with
    concentrated singular vectors has large :math:`\mu` and is not
    recoverable however many entries are sampled.
    """
    M = [[float(v) for v in r] for r in k.mat(A)]
    U, s, Vt = _svd(M)
    tol = max(len(M), len(M[0])) * (s[0] if len(s) else 0.0) * 1e-12
    r = int(rank) if rank is not None else sum(1 for v in s if v > tol)
    if r < 1:
        raise ValueError("meglt: the matrix is numerically zero")
    n1, n2 = len(M), len(M[0])
    mu_u = max(sum(U[i][j] ** 2 for j in range(r)) for i in range(n1))
    mu_v = max(sum(Vt[j][i] ** 2 for j in range(r)) for i in range(n2))
    return {"mu_row": n1 * mu_u / r, "mu_col": n2 * mu_v / r,
            "mu": max(n1 * mu_u / r, n2 * mu_v / r), "rank": r,
            "note": "large mu means concentrated singular vectors, "
                    "and then sampling reveals nothing"}


def sample_bound(n, r, C=1.0, exponent=1.2):
    r""":math:`C n^{1.2} r \log n`, or 1.25 to cover all ranks."""
    if exponent not in (1.2, 1.25):
        raise ValueError("meglt: the exponent must be 1.2 (moderate "
                         "rank) or 1.25 (all ranks), got %r"
                         % (exponent,))
    nn, rr = int(n), int(r)
    if nn < 2 or rr < 1:
        raise ValueError("meglt: need n >= 2 and r >= 1")
    m = float(C) * (nn ** float(exponent)) * rr * math.log(nn)
    return {"m": m, "fraction": m / float(nn * nn), "n": nn, "r": rr,
            "exponent": float(exponent),
            "note": "the 1.25 exponent holds for ALL ranks; 1.2 "
                    "assumes the rank is not too large"}


def svt(M, observed, tau=None, step=1.9, iters=200, tol=1e-6):
    r"""Singular value thresholding, projecting onto the observed
    entries."""
    A = [[float(v) for v in r] for r in k.mat(M)]
    n1, n2 = len(A), len(A[0])
    obs = set((int(i), int(j)) for i, j in observed)
    if not obs:
        raise ValueError("meglt: no entries were observed")
    t = float(tau) if tau is not None else 5.0 * math.sqrt(n1 * n2)
    Y = [[0.0] * n2 for _ in range(n1)]
    X = [[0.0] * n2 for _ in range(n1)]
    hist = []
    for _ in range(int(iters)):
        U, s, Vt = _svd(Y)
        sh = [max(0.0, v - t) for v in s]
        X = [[sum(U[i][q] * sh[q] * Vt[q][j]
                  for q in range(len(sh))) for j in range(n2)]
             for i in range(n1)]
        res = 0.0
        for (i, j) in obs:
            d = A[i][j] - X[i][j]
            res += d * d
            Y[i][j] += float(step) * d
        hist.append(math.sqrt(res))
        if hist[-1] < float(tol):
            break
    return RichResult(payload={
        "estimate": X, "X": X, "residual_history": hist,
        "final_residual": hist[-1], "tau": t,
        "n_observed": len(obs),
        "fraction_observed": len(obs) / float(n1 * n2),
        "nuclear_norm": nuclear_norm(X),
        "method": "singular value thresholding for the nuclear-norm "
                  "program; Candes & Recht (2009), Cai, Candes & Shen "
                  "(2010)",
    })


def relative_error(X, M):
    r""":math:`\|X - M\|_F/\|M\|_F`."""
    A = [[float(v) for v in r] for r in k.mat(M)]
    num = math.sqrt(sum((X[i][j] - A[i][j]) ** 2
                        for i in range(len(A))
                        for j in range(len(A[0]))))
    den = math.sqrt(sum(A[i][j] ** 2 for i in range(len(A))
                        for j in range(len(A[0]))))
    if den <= _EPS:
        raise ValueError("meglt: the reference matrix is zero")
    return num / den


def cheatsheet():
    return ("meglt: most low-rank matrices are recovered EXACTLY from "
            "m >= C n^1.2 r log n sampled entries -- 1.25 covers all "
            "ranks. Rank minimisation is NP-hard, so minimise the "
            "NUCLEAR NORM (sum of singular values), the rank's convex "
            "surrogate as l1 is for sparsity. INCOHERENCE is required, "
            "not decorative: e_1 e_1' is rank 1 and unrecoverable "
            "because nearly every sampled entry is zero. Solved by "
            "singular value thresholding.")


# compact alias per ledger/NAMING.md
matrixcompletion = svt
