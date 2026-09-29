# morie.fn -- function file (rootcoder007/morie)
"""Power weights W^p."""

from ._qpcore import ssum


def swpower(W, p=2):
    r"""Matrix power ``W^p`` of a spatial weights matrix (higher-order spatial operator).

    ``W^p`` weights the paths of length ``p`` in the neighbour graph; its
    trace appears in the series ``log|I - rho W| = -sum_p rho^p tr(W^p) / p``
    and in the spatial impacts ``(I - rho W)^{-1} = sum_p rho^p W^p``
    (LeSage and Pace 2009, ch. 4). ``p = 0`` gives the identity. Returns the
    matrix as lists.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press.

    Examples
    --------
    >>> swpower([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]], 2)
    [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
    """
    A = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    n = len(A)
    p = int(p)
    if p < 0:
        raise ValueError("p must be non-negative")
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(p):
        P = [[ssum(P[i][m] * A[m][j] for m in range(n)) for j in range(n)] for i in range(n)]
    return P


swpower_fn = swpower


def cheatsheet() -> str:
    return "swpower(W, p=2) -> matrix power W^p (paths of length p)."
