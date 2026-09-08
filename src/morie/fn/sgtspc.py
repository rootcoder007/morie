# morie.fn -- function file (rootcoder007/morie)
"""Spectrum of the normalised Laplacian."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["lapspec", "sgt_spectrum"]


def lapspec(W, tol=1e-10):
    """Eigenvalues of the normalised Laplacian, in increasing order.

    Two facts make this the diagnostic worth computing: the spectrum
    lies in [0, 2] for every graph, and the multiplicity of the
    eigenvalue 0 is the number of connected components.  Both are
    returned so a caller can see them rather than assume them.

    Formula: spectrum of Lcal = T^-1/2 (T - W) T^-1/2, lambda_0 = 0
             <= lambda_1 <= ... <= lambda_{n-1} <= 2

    Parameters
    ----------
    W : array-like, shape (n, n)
        Symmetric non-negative weight matrix.
    tol : float
        An eigenvalue below this counts as zero.

    Returns
    -------
    RichResult
        ``values`` (increasing), ``lambda1`` (smallest non-zero),
        ``n_components``, ``max_value``, ``n``.

    References
    ----------
    Chung (1997), Spectral Graph Theory, CBMS 92, Sections 1.2-1.3:
    the eigenvalues of Lcal lie in [0, 2] and the multiplicity of 0
    equals the number of connected components.  Fetched from the
    author's own copy of the chapter.
    """
    W = C.mat(W)
    n = len(W)
    if any(len(r) != n for r in W):
        raise ValueError("W must be square")
    d = [sum(W[i]) for i in range(n)]
    s = [0.0 if d[i] == 0.0 else d[i] ** -0.5 for i in range(n)]
    L = [[(d[i] - W[i][i]) if i == j else -W[i][j] for j in range(n)]
         for i in range(n)]
    Lc = [[s[i] * L[i][j] * s[j] for j in range(n)] for i in range(n)]
    vals, _ = C.eigsym(Lc)
    vals = list(reversed(vals))
    nz = [v for v in vals if v > tol]
    return RichResult(payload={
        "values": vals, "lambda1": nz[0] if nz else float("nan"),
        "n_components": sum(1 for v in vals if v <= tol),
        "max_value": vals[-1], "n": n,
        "method": "Spectrum of the normalised Laplacian"})


sgt_spectrum = lapspec


def cheatsheet():
    return "sgtspc: eigenvalues of Lcal in [0,2]; mult(0) = #components"
