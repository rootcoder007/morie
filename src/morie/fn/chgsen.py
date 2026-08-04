# morie.fn -- tail3 batch (rootcoder007/morie)
"""Change-of-variance function and change-of-variance sensitivity.

Source consulted: Hampel, F.R., Ronchetti, E.M., Rousseeuw, P.J. & Stahel,
W.A. (1986). *Robust Statistics: The Approach Based on Influence Functions*.
Wiley, section 2.5.  For a location M-estimator with score ``psi`` at a
symmetric model ``F`` the asymptotic variance is

    V(psi, F) = E[psi^2] / (E[psi'])^2

and the change-of-variance function, the derivative of ``V`` under
contamination, is

    CVF(x; psi, F) = V(psi, F) * ( 1 + psi(x)^2 / E[psi^2]
                                     - 2 psi'(x) / E[psi'] )

with the change-of-variance sensitivity kappa* = sup_x CVF(x) / V(psi, F).
``psi`` is V-robust at ``F`` exactly when kappa* is finite.

The expectations are taken with respect to the empirical distribution of the
supplied grid, so ``x`` should be a sample from ``F`` (or an equally spaced
grid weighted by ``F``, supplied through ``w``).
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["change_of_variance"]


def _deriv(psi, x):
    n = int(psi.size)
    d = [0.0] * n
    for i in range(n):
        if i == 0:
            d[i] = (float(psi[1]) - float(psi[0])) / (float(x[1]) - float(x[0]))
        elif i == n - 1:
            d[i] = (float(psi[n - 1]) - float(psi[n - 2])) / (float(x[n - 1]) - float(x[n - 2]))
        else:
            d[i] = (float(psi[i + 1]) - float(psi[i - 1])) / (float(x[i + 1]) - float(x[i - 1]))
    return np.asarray(d, dtype=float)


def change_of_variance(psi, x=None, dpsi=None, w=None):
    """Change-of-variance function and sensitivity of a location M-estimator.

    Parameters
    ----------
    psi : array-like
        Score function psi(x_i) on a grid.
    x : array-like, optional
        The grid.  Defaults to ``0, 1, ..., n-1``.
    dpsi : array-like, optional
        psi'(x_i).  If omitted, central differences on the grid are used.
    w : array-like, optional
        Probability weights for the expectations; defaults to uniform.

    Returns
    -------
    RichResult
        estimate (kappa*), V, cvfmax, cvfmin, vrobust, n, method.

    References
    ----------
    Hampel, Ronchetti, Rousseeuw & Stahel (1986), section 2.5.
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float)).ravel()
    n = int(psi.size)
    if x is None:
        xg = np.asarray([float(i) for i in range(n)], dtype=float)
    else:
        xg = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    dp = _deriv(psi, xg) if dpsi is None else np.atleast_1d(np.asarray(dpsi, dtype=float)).ravel()
    if w is None:
        ww = np.asarray([1.0 / n] * n, dtype=float)
    else:
        ww = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
        ww = ww / float(np.sum(ww))
    a = float(np.sum(ww * psi * psi))
    b = float(np.sum(ww * dp))
    v = a / (b * b) if b != 0.0 else float("nan")
    cvf = [v * (1.0 + float(psi[i]) ** 2 / a - 2.0 * float(dp[i]) / b) for i in range(n)]
    cvf = np.asarray(cvf, dtype=float)
    cvfmax = float(np.max(cvf))
    kappa = cvfmax / v if v != 0.0 else float("nan")
    return RichResult(
        payload={
            "estimate": kappa,
            "kappa_star": kappa,
            "V": v,
            "cvfmax": cvfmax,
            "cvfmin": float(np.min(cvf)),
            "cvf": cvf,
            "vrobust": bool(kappa < float("inf")),
            "n": n,
            "method": "Change-of-variance sensitivity (Hampel et al. 1986)",
        }
    )


# CANONICAL TEST
# >>> # psi(x) = x (the mean): V = E[x^2], CVF(x)/V = x^2/E[x^2] - 1
# >>> r = change_of_variance([-1.0, 0.0, 1.0], x=[-1.0, 0.0, 1.0], dpsi=[1.0, 1.0, 1.0])
# >>> assert abs(r["V"] - 2.0 / 3.0) < 1e-12


def cheatsheet():
    return "chgsen(psi, x, dpsi): change-of-variance function + kappa*."
