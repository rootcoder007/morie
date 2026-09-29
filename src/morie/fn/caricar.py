# morie.fn -- function file (rootcoder007/morie)
"""Intrinsic CAR (ICAR) log-density."""

import math

from ._containers import SpatialResult
from ._qpcore import ssum
from ._s03core import jacobi


def caricar(phi, W, tau=1.0, tol=1e-9):
    r"""Log-density of the intrinsic CAR (ICAR) prior, ``p(phi) ∝ tau^{r/2} exp(-tau/2 phi'Q phi)``.

    With ``W`` a symmetric non-negative adjacency, ``Q = D - W`` (``D`` the
    row sums) has rank ``r = n - c`` for ``c`` connected components, and the
    improper Gaussian (Besag, York and Mollie 1991; Rue and Held 2005, eq.
    3.18) has log-density

    ``log p = -r/2 log(2 pi) + r/2 log tau + 1/2 log|Q|* - tau/2 sum_{i~j}
    w_ij (phi_i - phi_j)^2``

    where ``|Q|*`` is the generalised determinant (product of the eigenvalues
    above ``tol * max``, from the Jacobi eigen-decomposition).

    References
    ----------
    Besag, J., York, J. and Mollie, A. (1991). Bayesian image restoration,
    with two applications in spatial statistics. *Ann. Inst. Statist.
    Math.* 43, 1-20.
    Rue, H. and Held, L. (2005). *Gaussian Markov Random Fields*. Chapman and
    Hall/CRC.

    Examples
    --------
    >>> W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> round(caricar([0.5, -0.2, -0.3], W).statistic, 12)
    -1.538570922075
    """
    x = [float(v) for v in (phi.tolist() if hasattr(phi, "tolist") else phi)]
    Wm = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    n = len(x)
    if any(Wm[i][j] != Wm[j][i] for i in range(n) for j in range(n)):
        raise ValueError("W must be symmetric")
    Q = [[(ssum(Wm[i]) if i == j else 0.0) - Wm[i][j] for j in range(n)] for i in range(n)]
    vals, _ = jacobi(Q)
    top = max(vals)
    pos = [v for v in vals if v > tol * top]
    r = len(pos)
    quad = ssum(Wm[i][j] * (x[i] - x[j]) ** 2 for i in range(n) for j in range(i + 1, n))
    t = float(tau)
    logp = -0.5 * r * math.log(2.0 * math.pi) + 0.5 * r * math.log(t) + 0.5 * ssum(math.log(v) for v in pos)
    logp -= 0.5 * t * quad
    return SpatialResult(name="caricar", statistic=logp, extra={"rank": r, "quadratic_form": quad, "tau": t})


caricar_fn = caricar


def cheatsheet() -> str:
    return "caricar(phi, W, tau=1) -> ICAR log-density with rank n - c and generalised determinant of D - W."
