# morie.fn -- function file (rootcoder007/morie)
"""CAR Besag-York-Mollié (BYM) variance components."""

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._s03core import jacobi


def _profile(t2, lam, phi):
    a = [phi / lv + 1.0 - phi for lv in lam]
    m = len(lam)
    s = ssum(t2[j] / a[j] for j in range(m))
    ll = -0.5 * m * (math.log(2.0 * math.pi * s / m) + 1.0) - 0.5 * ssum(math.log(v) for v in a)
    ds = -ssum(t2[j] * (1.0 / lam[j] - 1.0) / (a[j] * a[j]) for j in range(m))
    score = -0.5 * m * ds / s - 0.5 * ssum((1.0 / lam[j] - 1.0) / a[j] for j in range(m))
    return ll, score, s / m


def carbym(y, W, tol=1e-9):
    r"""REML variance components of the Gaussian Besag-York-Mollie (BYM) convolution model.

    ``y = mu 1 + u + v`` with ``u`` an intrinsic CAR field of precision
    ``Q / sigma_u^2`` (``Q = D - W``, ``W`` a symmetric adjacency) and ``v ~
    N(0, sigma_v^2 I)`` (Besag, York and Mollie 1991). In the eigenbasis ``Q
    = V diag(lambda) V'`` the contrasts ``t_j = v_j'y`` for ``lambda_j > 0``
    are independent ``N(0, sigma_u^2 / lambda_j + sigma_v^2)`` and free of
    ``mu``, so their likelihood is the REML likelihood. Writing ``sigma_u^2 =
    s phi`` and ``sigma_v^2 = s (1 - phi)``, ``s`` is profiled out and the
    REML score in ``phi`` is solved by bisection on ``[0, 1]`` (boundary
    solutions are reported when the score does not change sign).

    Returns ``sigma2_spatial``, ``sigma2_unstructured``, ``phi`` (spatial
    share of ``sigma_u^2 + sigma_v^2``), ``mu`` (the component means are
    GLS-free here: the plain mean for a connected graph), ``reml_loglik``.

    References
    ----------
    Besag, J., York, J. and Mollie, A. (1991). Bayesian image restoration,
    with two applications in spatial statistics. *Ann. Inst. Statist.
    Math.* 43, 1-20.
    Patterson, H. D. and Thompson, R. (1971). Recovery of inter-block
    information when block sizes are unequal. *Biometrika* 58, 545-554.

    Examples
    --------
    >>> import math
    >>> W = [[1.0 if abs(i // 4 - j // 4) + abs(i % 4 - j % 4) == 1 else 0.0 for j in range(16)] for i in range(16)]
    >>> y = [0.5 * (i // 4) + 0.4 * (i % 4) + 0.8 * math.sin(2.3 * i) for i in range(16)]
    >>> round(carbym(y, W)["phi"], 9)
    0.64281446
    """
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    Wm = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    n = len(yv)
    if any(Wm[i][j] != Wm[j][i] for i in range(n) for j in range(n)):
        raise ValueError("W must be symmetric")
    Q = [[(ssum(Wm[i]) if i == j else 0.0) - Wm[i][j] for j in range(n)] for i in range(n)]
    vals, vecs = jacobi(Q)
    top = max(vals)
    keep = [j for j in range(n) if vals[j] > tol * top]
    lam = [vals[j] for j in keep]
    t2 = [ssum(vecs[i][j] * yv[i] for i in range(n)) ** 2 for j in keep]
    _, g0, _ = _profile(t2, lam, 0.0)
    _, g1, _ = _profile(t2, lam, 1.0)
    if g0 <= 0.0:
        phi = 0.0
    elif g1 >= 0.0:
        phi = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _profile(t2, lam, mid)[1] > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-15:
                break
        phi = 0.5 * (lo + hi)
    ll, _, s = _profile(t2, lam, phi)
    return RichResult(
        payload={
            "sigma2_spatial": s * phi,
            "sigma2_unstructured": s * (1.0 - phi),
            "phi": phi,
            "mu": ssum(yv) / n,
            "reml_loglik": ll,
            "rank": len(lam),
        }
    )


carbym_fn = carbym


def cheatsheet() -> str:
    return "carbym(y, W) -> REML sigma_u^2 (ICAR) and sigma_v^2 (iid) of the Gaussian BYM model."
