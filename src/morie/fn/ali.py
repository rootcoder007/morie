# morie.fn -- tail3 batch (rootcoder007/morie)
"""Ali-Mikhail-Haq copula.

Source consulted: Ali, M.M., Mikhail, N.N. & Haq, M.S. (1978). A class of
bivariate distributions including the bivariate logistic.  *Journal of
Multivariate Analysis* 8(3), 405-412.  The copula is

    C(u, v) = u v / (1 - theta (1 - u)(1 - v)),      theta in [-1, 1)

an Archimedean copula with generator ``log((1 - theta(1 - t))/t)``.  Its
density is obtained by differentiating C twice, and Kendall's tau is

    tau = (3 theta - 2)/(3 theta) - 2 (1 - theta)^2 log(1 - theta)/(3 theta^2)

which restricts tau to roughly [-0.1817, 1/3]; theta = 0 gives the
independence copula and tau = 0.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ali_mikhail_haq_copula"]


def _amh_tau(theta):
    t = float(theta)
    if t == 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0 / 3.0
    return (3.0 * t - 2.0) / (3.0 * t) - 2.0 * (1.0 - t) ** 2 * float(np.log(1.0 - t)) / (3.0 * t * t)


def ali_mikhail_haq_copula(u, v, theta=0.0):
    """Ali-Mikhail-Haq copula, density and Kendall's tau.

    Parameters
    ----------
    u, v : array-like
        Values in (0, 1).
    theta : float
        Association parameter in [-1, 1).

    Returns
    -------
    RichResult
        estimate (mean C(u, v)), tau, cdf, density, loglik, theta, n, method.

    References
    ----------
    Ali, Mikhail & Haq (1978), J. Multivariate Analysis 8(3), 405-412.
    """
    uu = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    vv = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    n = int(min(uu.size, vv.size))
    th = float(theta)
    cdf = []
    dens = []
    for i in range(n):
        a = float(uu[i])
        b = float(vv[i])
        d = 1.0 - th * (1.0 - a) * (1.0 - b)
        cdf.append(a * b / d)
        num = 1.0 - th + 2.0 * th * a * b / d - th * (1.0 - a) * (1.0 - b) / d
        dens.append(num / (d * d))
    cdfa = np.asarray(cdf, dtype=float)
    densa = np.asarray(dens, dtype=float)
    loglik = float(np.sum(np.log(densa)))
    return RichResult(
        payload={
            "estimate": float(np.mean(cdfa)),
            "cdf": cdfa,
            "density": densa,
            "loglik": loglik,
            "tau": float(_amh_tau(th)),
            "theta": th,
            "n": n,
            "method": "Ali-Mikhail-Haq copula (Ali, Mikhail & Haq 1978)",
        }
    )


# CANONICAL TEST
# >>> # theta = 0 is the independence copula: C(u, v) = u v, density 1, tau 0
# >>> r = ali_mikhail_haq_copula([0.5], [0.4], 0.0)
# >>> assert abs(r["estimate"] - 0.2) < 1e-12
# >>> assert abs(r["loglik"]) < 1e-12
# >>> assert abs(r["tau"]) < 1e-12


def cheatsheet():
    return "ali(u, v, theta): Ali-Mikhail-Haq copula, density, Kendall tau."
