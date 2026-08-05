"""Combined spatial autoregressive lag + autoregressive error (SARAR)."""

import math as _math

from . import _array_core as np
from ._schab_rho import safe_search_interval

from ._richresult import RichResult

__all__ = ["spatial_ar_combined"]

_LEVELS = 7
_NGRID = 21


def _grid_refine(negll, lo1, hi1, lo2, hi2):
    """Deterministic nested-grid minimiser on a rectangle.

    Both language arms run the identical arithmetic -- a fixed
    ``_NGRID x _NGRID`` sweep, then ``_LEVELS`` zooms each shrinking the
    box by ten -- so they land on the same numbers rather than on two
    different optimiser trajectories.  Final resolution is
    ``width / (20 * 10^(_LEVELS-1))``.
    """
    a1, b1, a2, b2 = lo1, hi1, lo2, hi2
    best = (float("inf"), a1, a2)
    for _ in range(_LEVELS):
        s1 = (b1 - a1) / (_NGRID - 1)
        s2 = (b2 - a2) / (_NGRID - 1)
        best = (float("inf"), a1, a2)
        for i in range(_NGRID):
            u = a1 + i * s1
            for j in range(_NGRID):
                v = a2 + j * s2
                f = negll(u, v)
                if f < best[0]:
                    best = (f, u, v)
        a1n, b1n = best[1] - s1, best[1] + s1
        a2n, b2n = best[2] - s2, best[2] + s2
        a1, b1 = (a1n if a1n > lo1 else lo1), (b1n if b1n < hi1 else hi1)
        a2, b2 = (a2n if a2n > lo2 else lo2), (b2n if b2n < hi2 else hi2)
    return best[1], best[2], best[0]


def spatial_ar_combined(y, X, W1, W2):
    """
    Combined SAR-SEM ("SARAR", "SAC") model, by concentrated ML.

    Model (Kelejian & Prucha 1998, eqs. (1)-(2), p. 101)::

        y = rho W1 y + X beta + u
        u = lam W2 u + eps,        eps ~ N(0, sigma2 I)

    Writing ``A = I - rho W1`` and ``B = I - lam W2``, the transformed
    system ``B A y = B X beta + eps`` is spherical, so for fixed
    ``(rho, lam)`` the remaining parameters are ordinary least squares::

        ystar = B A y,   Xstar = B X
        beta(rho, lam)   = (Xstar' Xstar)^{-1} Xstar' ystar
        sigma2(rho, lam) = e'e / n,   e = ystar - Xstar beta

    and the profile log-likelihood is::

        ll(rho, lam) = -n/2 log(2 pi sigma2) + log|A| + log|B| - n/2

    which is maximised over the admissible rectangle.  The admissible
    range of each parameter is the eigenvalue interval
    ``(1/theta_min, 1/theta_max)`` on which ``|I - t W|`` stays positive
    (Schabenberger & Gotway 2005, eq. 6.48, p. 340) -- not a hardcoded
    ``(-0.99, 0.99)``, which is wrong for a raw adjacency.

    Setting ``lam = 0`` recovers the SAR lag model (``sarla``) and
    ``rho = 0`` recovers the SAR error model.  Passing the same matrix
    for ``W1`` and ``W2`` gives the SAC model (see ``sacmod``).

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, p)
        Design matrix; the intercept must be explicit.
    W1 : array-like, shape (n, n)
        Weights for the autoregressive lag in the response.
    W2 : array-like, shape (n, n)
        Weights for the autoregressive disturbance.

    Returns
    -------
    RichResult
        Payload keys: ``estimate``, ``se``, ``rho``, ``lambda``,
        ``sigma2``, ``loglik``, ``n``, ``method``.

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (1998). A generalized spatial
    two-stage least squares procedure for estimating a spatial
    autoregressive model with autoregressive disturbances. The Journal of
    Real Estate Finance and Economics 17(1), 99-121.
    doi:10.1023/A:1007707430416.
    Anselin, L. (1988). Spatial Econometrics: Methods and Models.
    Schabenberger, O. and Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis, Sec. 6.2.2, pp. 335-341.
    """
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim == 1:
        Xm = Xm.reshape(-1, 1)
    yv = np.asarray(y, dtype=float).ravel()
    A1 = np.asarray(W1, dtype=float)
    A2 = np.asarray(W2, dtype=float)
    n, p = int(Xm.shape[0]), int(Xm.shape[1])
    if yv.size != n or A1.shape != (n, n) or A2.shape != (n, n):
        raise ValueError("shape mismatch among y, X, W1, W2")
    if n <= p:
        raise ValueError("need more observations than columns of X")
    I = np.eye(n)

    def parts(rho, lam):
        A = I - rho * A1
        B = I - lam * A2
        ystar = B @ (A @ yv)
        Xstar = B @ Xm
        G = Xstar.T @ Xstar
        beta = np.linalg.solve(G, Xstar.T @ ystar)
        e = ystar - Xstar @ beta
        return A, B, G, beta, e

    def negll(rho, lam):
        try:
            A, B, G, beta, e = parts(rho, lam)
        except Exception:
            return 1e12
        s2 = float(e @ e) / n
        if not (s2 > 0.0):
            return 1e12
        sa, la = np.linalg.slogdet(A)
        sb, lb = np.linalg.slogdet(B)
        if sa <= 0 or sb <= 0:
            return 1e12
        return 0.5 * n * _math.log(2.0 * _math.pi * s2) - la - lb + 0.5 * n

    lo1, hi1 = safe_search_interval(A1, "identity")
    lo2, hi2 = safe_search_interval(A2, "identity")
    rho, lam, fmin = _grid_refine(negll, lo1, hi1, lo2, hi2)

    A, B, G, beta, e = parts(rho, lam)
    sigma2 = float(e @ e) / max(n - p, 1)
    cov = sigma2 * np.linalg.inv(G)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return RichResult(
        payload={
            "estimate": beta.tolist(),
            "se": se.tolist(),
            "rho": float(rho),
            "lambda": float(lam),
            "sigma2": sigma2,
            "loglik": -float(fmin),
            "n": n,
            "method": "SARAR (SAR lag + SAR error) by concentrated ML",
        }
    )


def cheatsheet():
    return "sarmix: combined SAR lag + SAR error (SARAR) model"


# compact alias per ledger/NAMING.md
spatialarcombined = spatial_ar_combined
