# morie.fn -- function file (rootcoder007/morie)
"""Arc-cosine kernel matrix."""

from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['arckern', 'arc_cosine_kernel']


def arckern(X, Z=None, depth=1):
    """Arc-cosine kernel matrix.

    Formula: K(x_i, x_j) = (1/pi) * ||x_i|| * ||x_j|| * [sin(theta) + (pi - theta) cos(theta)],  theta = angle(x_i, x_j)

    Parameters
    ----------
    X : array-like, shape (n, p)
        One record per row.
    Z : array-like or None
        Second set of records; None gives the square Gram matrix of X.
    depth : int
        Number of times the kernel is composed with itself.

    Returns
    -------
    RichResult
        ``K``, ``depth``, ``n``, ``m``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 8, Sect. 8.2.2, the arc-cosine kernel named on p. 252 and developed later in the chapter; the implementation delegates to the chapter-8 arc-cosine kernel already verified against the book for this shelf.  Read from the chapter PDF, not recalled.
    """
    out = G.arccos_kernel(X, Z=Z, depth=int(depth))
    K = out["K"] if isinstance(out, dict) else out
    return RichResult(payload={
        "K": K, "depth": int(depth), "n": len(K), "m": len(K[0]),
        "method": "Arc-cosine kernel, MVSML Chap. 8"})


arc_cosine_kernel = arckern


def cheatsheet():
    return 'arckn: Arc-cosine kernel matrix.'
