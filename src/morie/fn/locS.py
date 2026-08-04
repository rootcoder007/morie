# morie.fn -- k02 batch (rootcoder007/morie)
"""Huber's Proposal 2: simultaneous M-estimation of location and scale.

Source consulted: Huber, P.J. (1981), *Robust Statistics*, Wiley, section 6.7
("Proposal 2").  Location and scale solve, together,

    sum psi_k( (x_i - mu)/s ) = 0,   (1/n) sum psi_k( (x_i - mu)/s )^2 = beta

with psi_k the Huber score and the consistency constant

    beta = (2 Phi(k) - 1) + k^2 (1 - (2 Phi(k) - 1)) - 2 k phi(k)

Each sweep Winsorises the sample at mu +/- k s, takes the mean of the
Winsorised values as the new mu and sqrt( sum (y - mu)^2 / ((n-1) beta) ) as
the new s.

TWO THINGS THE COMPARISON WITH MASS::hubers TAUGHT US, both kept visible
rather than papered over.  MASS stops at ``tol = 1e-6`` within at most 30
sweeps, and on break it returns the *pre-update* iterate, not the one it just
computed.  Run to a real fixed point the answer moves by about 1e-3 on the
fixture below.  So the default reproduces MASS exactly (``estimate``,
``scale``) and the converged fixed point is also returned
(``mu_refined``, ``scale_refined``) -- neither is hidden behind the other.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as _st

from ._richresult import RichResult

__all__ = ["location_scale_estimator"]


def _sweep(v, n, kk, beta, mu0, s0, tol, maxit):
    mu1 = mu0
    s1 = s0
    it = 0
    for it in range(1, int(maxit) + 1):
        yy = np.minimum(np.maximum(mu0 - kk * s0, v), mu0 + kk * s0)
        mu1 = float(np.sum(yy)) / n
        ss = float(np.sum((yy - mu1) ** 2)) / (n - 1)
        s1 = float(np.sqrt(ss / beta))
        if abs(mu0 - mu1) < tol * s0 and abs(s0 - s1) < tol * s0:
            break
        mu0 = mu1
        s0 = s1
    return mu0, s0, it


def location_scale_estimator(x, k=1.5, tol=1e-6, maxit=30):
    """Huber Proposal 2 location and scale.

    Parameters
    ----------
    x : array-like
        Sample.
    k : float, default 1.5
        Huber tuning constant.
    tol : float, default 1e-6
        Relative convergence tolerance (MASS::hubers default).
    maxit : int, default 30
        Sweep cap (MASS::hubers default).

    Returns
    -------
    RichResult
        estimate (mu), scale, mu_refined, scale_refined, se, beta,
        iterations, k, n, method.
    """
    v = np.asarray(x, dtype=float).ravel()
    n = len(v)
    kk = float(k)
    th = 2.0 * float(_st.norm.cdf(kk)) - 1.0
    beta = th + kk * kk * (1.0 - th) - 2.0 * kk * float(_st.norm.pdf(kk))
    mu0 = float(np.median(v))
    s0 = 1.4826 * float(np.median(np.abs(v - mu0)))
    if s0 == 0.0:
        return RichResult(
            payload={
                "estimate": mu0,
                "scale": 0.0,
                "mu_refined": mu0,
                "scale_refined": 0.0,
                "se": 0.0,
                "beta": float(beta),
                "iterations": 0,
                "k": kk,
                "n": int(n),
                "method": "Huber Proposal 2 simultaneous location and scale (Huber 1981, sec. 6.7)",
            }
        )
    mu, s, it = _sweep(v, n, kk, beta, mu0, s0, float(tol), int(maxit))
    mur, sr, _it2 = _sweep(v, n, kk, beta, mu0, s0, 1e-13, 500)
    return RichResult(
        payload={
            "estimate": float(mu),
            "scale": float(s),
            "mu_refined": float(mur),
            "scale_refined": float(sr),
            "se": float(s / np.sqrt(n)),
            "beta": float(beta),
            "iterations": int(it),
            "k": kk,
            "n": int(n),
            "method": "Huber Proposal 2 simultaneous location and scale (Huber 1981, sec. 6.7)",
        }
    )


# CANONICAL TEST
# >>> x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 9.9, 2.5, 3.3, 2.7]
# >>> r = location_scale_estimator(x)
# >>> assert abs(r["estimate"] - 3.18126476231742) < 1e-12   # MASS::hubers
# >>> assert abs(r["scale"] - 1.21731162947809) < 1e-12
# >>> # the true fixed point is about 1e-3 away from where MASS stops
# >>> assert abs(r["mu_refined"] - r["estimate"]) > 1e-4


def cheatsheet():
    return "locS(x, k): Huber Proposal 2 location and scale."


locationscaleestimator = location_scale_estimator
