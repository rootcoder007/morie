# morie.fn -- function file (rootcoder007/morie)
"""CAR variance-covariance matrix."""

from ._qpcore import inverse, ssum
from ._richresult import RichResult


def _mat(A):
    return [[float(v) for v in r] for r in (A.tolist() if hasattr(A, "tolist") else A)]


def carvar(W, rho, sigma2, m=None):
    r"""Covariance matrix of a Gaussian conditional autoregressive (CAR) field.

    The CAR full conditionals ``x_i | x_-i ~ N(rho sum_j w_ij x_j, sigma2
    m_i)`` define, by Brook's lemma / Hammersley-Clifford, the joint
    ``N(0, Sigma)`` with ``Sigma = sigma2 (I - rho W)^{-1} M``, ``M =
    diag(m)`` (default ``m_i = 1``), provided ``(I - rho W)^{-1} M`` is
    symmetric positive definite (Besag 1974; Cressie 1993, eq. 6.3.4). For
    the weighted form pass ``W = D^{-1} A`` and ``m = 1 / d``.

    References
    ----------
    Besag, J. (1974). Spatial interaction and the statistical analysis of
    lattice systems. *J. R. Stat. Soc. B* 36, 192-236.
    Cressie, N. (1993). *Statistics for Spatial Data*, rev. ed. Wiley.

    Examples
    --------
    >>> W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> r = carvar(W, 0.4, 2.0)
    >>> round(r["covariance"][0][0], 12)
    2.470588235294
    """
    Wm = _mat(W)
    n = len(Wm)
    mv = [1.0] * n if m is None else [float(v) for v in m]
    A = [[(1.0 if i == j else 0.0) - float(rho) * Wm[i][j] for j in range(n)] for i in range(n)]
    Ai = inverse(A)
    S = [[float(sigma2) * Ai[i][j] * mv[j] for j in range(n)] for i in range(n)]
    asym = max(abs(S[i][j] - S[j][i]) for i in range(n) for j in range(n))
    if asym > 1e-8 * max(abs(v) for r in S for v in r):
        raise ValueError("(I - rho W)^{-1} M is not symmetric: not a valid CAR specification")
    return RichResult(
        payload={
            "covariance": S,
            "marginal_variance": [S[i][i] for i in range(n)],
            "mean_marginal_variance": ssum(S[i][i] for i in range(n)) / n,
        }
    )


carvar_fn = carvar


def cheatsheet() -> str:
    return "carvar(W, rho, sigma2, m=None) -> Sigma = sigma2 (I - rho W)^{-1} diag(m) of a CAR field."
