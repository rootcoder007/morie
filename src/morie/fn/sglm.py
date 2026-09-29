"""Spatial generalized linear model (Gaussian-link spatial GLM)."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from .krgsys import kriging_covariance
from .sglmm import glmm_laplace_fit
from .vgmods import likfit

__all__ = ["spatial_glm"]


def _mat(A):
    A = A.tolist() if hasattr(A, "tolist") else A
    return [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in A]


def spatial_glm(x, y, coords, family: str = "gaussian", model: str = "Exp"):
    r"""Spatial generalized linear model (Gaussian-link spatial GLM).

    ``g(E y_i) = x_i'beta + S(s_i)`` with a stationary Gaussian field ``S``
    of covariance ``sigma^2 rho(d / phi)`` (``model`` ``Exp``, ``Gau`` or
    ``Sph``; Schabenberger and Gotway 2005, ch. 5 and 6; Diggle, Tawn and
    Moyeed 1998).

    - ``family="gaussian"`` (identity link): the spatial linear model ``y =
      X beta + S + e``, ``e ~ N(0, tau^2)``, by maximum likelihood with
      ``beta`` and ``sigma^2`` profiled out and ``(phi, tau^2/sigma^2)``
      searched (:func:`morie.fn.vgmods.likfit`, as ``geoR::likfit``); the
      standard errors of ``beta`` are ``sqrt(diag((X'V^{-1}X)^{-1}))`` at the
      estimates, ``V = sigma^2 R + tau^2 I``.
    - ``family="poisson"`` (log) or ``"binomial"`` (logit): the spatial
      GLMM by maximising the Laplace approximation of the marginal
      likelihood (:func:`morie.fn.sglmm.glmm_laplace_fit`).

    The range search starts at a third of the largest distance.

    Parameters
    ----------
    x : array-like, shape (n, p) or (n,)
        Design matrix (include an intercept column for an intercept).
    y : array-like, shape (n,)
        Responses.
    coords : array-like, shape (n, 2) or (n,)
        Locations (1-D coordinates are placed on a line).
    family : str
        ``"gaussian"``, ``"poisson"`` or ``"binomial"``.
    model : str
        Correlation model.

    Returns
    -------
    RichResult
        ``estimate`` (``beta``), ``se`` (Gaussian only), ``sigma2``,
        ``phi``, ``tau2``, ``loglik``, ``n``, ``method``.

    References
    ----------
    Schabenberger, O. and Gotway, C. A. (2005). *Statistical Methods for Spatial Data Analysis*.
    Chapman and Hall/CRC, ch. 5-6.

    Diggle, P. J., Tawn, J. A. and Moyeed, R. A. (1998). Model-based geostatistics. *Applied
    Statistics*, 47(3), 299-350.

    Examples
    --------
    >>> P = [(float(i % 4), float(i // 4)) for i in range(12)]
    >>> X = [[1.0, p[0]] for p in P]
    >>> y = [1.1, 1.9, 3.2, 3.8, 1.3, 2.2, 2.9, 4.1, 0.8, 2.1, 3.0, 4.2]
    >>> r = spatial_glm(X, y, P)
    >>> [round(b, 4) for b in r["estimate"]]
    [1.07, 0.9867]
    """
    X = _mat(x)
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    P = _mat(coords)
    if len(P[0]) == 1:
        P = [[r[0], 0.0] for r in P]
    n, p = len(X), len(X[0])
    if len(yv) != n or len(P) != n:
        raise ValueError("shape mismatch among x, y, coords")
    D = [[math.dist(P[i], P[j]) for j in range(n)] for i in range(n)]
    dmax = max(max(r) for r in D)
    if family == "gaussian":
        f = likfit(yv, P, {"model": model, "range": dmax / 3.0}, X=X)
        comp = {"model": model, "psill": f["psill"], "range": f["range"]}
        V = [[kriging_covariance(D[i][j], comp) + (f["nugget"] if i == j else 0.0) for j in range(n)] for i in range(n)]
        Vi = [[float(v) for v in r] for r in inverse(V)]
        ViX = [[ssum(Vi[i][m] * X[m][k] for m in range(n)) for k in range(p)] for i in range(n)]
        C = [
            [float(v) for v in r]
            for r in inverse([[ssum(X[i][a] * ViX[i][b] for i in range(n)) for b in range(p)] for a in range(p)])
        ]
        return RichResult(
            payload={
                "estimate": list(f["beta"]),
                "se": [math.sqrt(C[k][k]) for k in range(p)],
                "sigma2": f["psill"],
                "phi": f["range"],
                "tau2": f["nugget"],
                "loglik": f["loglik"],
                "n": n,
                "method": f"Spatial linear model, {model} covariance with nugget, ML (likfit)",
            }
        )
    if family not in ("poisson", "binomial"):
        raise ValueError("family must be 'gaussian', 'poisson' or 'binomial'")
    g = glmm_laplace_fit(yv, X, family=family, coords=P, model=model)
    return RichResult(
        payload={
            "estimate": list(g["beta"]),
            "se": None,
            "sigma2": g["sigma2"],
            "phi": g["range"],
            "tau2": 0.0,
            "loglik": g["loglik"],
            "n": n,
            "method": f"Spatial GLMM ({family}), Laplace approximation",
        }
    )


def cheatsheet():
    return "sglm: spatial linear model (likfit) or spatial GLMM (Laplace) by family"


# compact alias per ledger/NAMING.md
spatialglm = spatial_glm
