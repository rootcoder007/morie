# morie.fn -- function file (rootcoder007/morie)
"""Laplacian eigenvectors and the Fiedler vector."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["lapeig", "laplacian_eigen"]


def lapeig(W, k=2, tol=1e-10):
    """Eigenvectors of the normalised Laplacian for the k smallest values.

    The eigenvector for the smallest non-zero eigenvalue -- the Fiedler
    vector -- is the one that carries the partition; its sign pattern
    is the spectral bisection Chung's Theorem 2.2 bounds.  Eigenvectors
    are sign-fixed on their largest-magnitude entry, without which the
    two language arms would disagree on an arbitrary sign and no
    parity test could ever pass.  Under a repeated eigenvalue the
    individual vectors remain arbitrary within their eigenspace; only
    the eigenvalues and the projector are well defined.

    Formula: Lcal u = lambda u, lambda_0 <= ... <= lambda_{k-1};
             Fiedler vector = u for the smallest lambda > tol

    Parameters
    ----------
    W : array-like, shape (n, n)
        Symmetric non-negative weight matrix.
    k : int
        Number of eigenpairs returned (1 <= k <= n).
    tol : float
        An eigenvalue below this counts as zero.

    Returns
    -------
    RichResult
        ``values``, ``vectors`` (n x k, column j is the j-th
        eigenvector), ``fiedler``, ``lambda1``, ``n_components``,
        ``n``, ``k``.

    References
    ----------
    Chung (1997), Spectral Graph Theory, CBMS 92, Section 1.2, which
    writes Lcal as an operator on functions g : V -> R and works with
    its harmonic eigenfunctions.  Fetched from the author's own copy of
    the chapter.
    """
    W = C.mat(W)
    n = len(W)
    if any(len(r) != n for r in W):
        raise ValueError("W must be square")
    k = int(k)
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n")
    d = [sum(W[i]) for i in range(n)]
    s = [0.0 if d[i] == 0.0 else d[i] ** -0.5 for i in range(n)]
    L = [[(d[i] - W[i][i]) if i == j else -W[i][j] for j in range(n)]
         for i in range(n)]
    Lc = [[s[i] * L[i][j] * s[j] for j in range(n)] for i in range(n)]
    vals, vecs = C.eigsym(Lc)
    order = list(reversed(range(n)))
    vals = [vals[i] for i in order]
    V = [[vecs[r][order[j]] for j in range(k)] for r in range(n)]
    nzi = [i for i in range(n) if vals[i] > tol]
    fied = ([vecs[r][order[nzi[0]]] for r in range(n)] if nzi
            else [0.0] * n)
    return RichResult(payload={
        "values": vals[:k], "vectors": V, "fiedler": fied,
        "lambda1": vals[nzi[0]] if nzi else float("nan"),
        "n_components": sum(1 for v in vals if v <= tol), "n": n, "k": k,
        "method": "Normalised-Laplacian eigenvectors (sign-fixed)"})


laplacian_eigen = lapeig


def cheatsheet():
    return "laplmo: k smallest eigenpairs of Lcal, sign-fixed; Fiedler vector"
