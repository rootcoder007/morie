# morie.fn -- function file (rootcoder007/morie)
"""ARMA(p, q) fitting by conditional least squares."""

import numpy as np

from ._richresult import RichResult

__all__ = ["arma_model"]


def arma_model(x, p=1, q=1, max_iter=200, tol=1e-10):
    r"""Fit an ARMA(p, q) model and check that it is usable.

    .. math::
       X_t = c + \sum_{i=1}^{p}\phi_i X_{t-i}
               + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}
               + \varepsilon_t

    Estimated by conditional least squares: the unobserved
    :math:`\varepsilon_{t-j}` are replaced by residuals from the
    current fit and the whole thing iterated. This is the Hannan-Rissanen
    idea and it converges quickly when the model is invertible.

    Two properties decide whether the fit means anything, and both are
    computed from the fitted roots rather than assumed. STATIONARITY
    requires every root of :math:`1 - \phi_1 z - \cdots - \phi_p z^p`
    to lie outside the unit circle; without it the process has no fixed
    mean and forecasts diverge. INVERTIBILITY requires the same of the
    MA polynomial; without it the model cannot be written as an
    infinite autoregression and the parameters are not identified --
    an MA(1) with :math:`\theta` and one with :math:`1/\theta` give
    the SAME autocorrelations, so the data cannot distinguish them.

    Parameters
    ----------
    x : array-like, shape (n,)
    p, q : int
        AR and MA orders.
    max_iter, tol : int, float

    Returns
    -------
    RichResult
        ``ar``, ``ma``, ``intercept``, ``sigma2``, ``residuals``,
        ``aic``, ``bic``, ``stationary``, ``invertible``,
        ``ar_roots``, ``ma_roots``.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 15,
    on ARMA for time series. Box and Jenkins (1970).
    Hannan and Rissanen (1982), *Biometrika* 69:81-94.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> e = rng.normal(size=400)
    >>> z = np.zeros(400)
    >>> for t in range(1, 400):
    ...     z[t] = 0.6 * z[t - 1] + e[t]
    >>> out = arma_model(z, p=1, q=0)
    >>> bool(abs(out["ar"][0] - 0.6) < 0.15)
    True
    """
    v = np.asarray(x, dtype=float).ravel()
    n = v.size
    p, q = int(p), int(q)
    if p < 0 or q < 0:
        raise ValueError("p and q must be non-negative.")
    if p + q == 0:
        raise ValueError("need at least one AR or MA term.")
    if n <= p + q + 2:
        raise ValueError(
            "series of length %d is too short for ARMA(%d, %d)." % (n, p, q)
        )

    m = max(p, q)
    eps = np.zeros(n)
    coef = np.zeros(1 + p + q)
    for _ in range(int(max_iter)):
        rows, targ = [], []
        for t in range(m, n):
            r = [1.0]
            r += [v[t - i] for i in range(1, p + 1)]
            r += [eps[t - j] for j in range(1, q + 1)]
            rows.append(r)
            targ.append(v[t])
        A = np.asarray(rows)
        b = np.asarray(targ)
        new, *_ = np.linalg.lstsq(A, b, rcond=None)
        delta = float(np.max(np.abs(new - coef)))
        coef = new
        fitted = A @ coef
        eps = np.zeros(n)
        eps[m:] = b - fitted
        if delta < tol:
            break

    c = float(coef[0])
    phi = coef[1:1 + p]
    theta = coef[1 + p:]
    resid = eps[m:]
    k = 1 + p + q
    s2 = float(np.sum(resid ** 2) / max(resid.size - k, 1))
    ll = -0.5 * resid.size * (np.log(2 * np.pi * s2) + 1.0)
    ar_roots = np.roots(np.concatenate([[1.0], -phi])) if p else np.array([])
    ma_roots = np.roots(np.concatenate([[1.0], theta])) if q else np.array([])
    stat = bool(np.all(np.abs(ar_roots) > 1.0)) if p else True
    inv = bool(np.all(np.abs(ma_roots) > 1.0)) if q else True
    return RichResult(
        payload={
            "estimate": np.concatenate([[c], phi, theta]),
            "intercept": c,
            "ar": phi,
            "ma": theta,
            "sigma2": s2,
            "residuals": resid,
            "loglik": float(ll),
            "aic": float(-2 * ll + 2 * k),
            "bic": float(-2 * ll + k * np.log(resid.size)),
            "ar_roots": ar_roots,
            "ma_roots": ma_roots,
            "stationary": stat,
            "invertible": inv,
            "stationarity_note": (
                None if stat else
                "an AR root is inside the unit circle: the process has no "
                "fixed mean and forecasts will diverge"
            ),
            "invertibility_note": (
                None if inv else
                "an MA root is inside the unit circle: the model has no "
                "infinite-AR representation and theta is not identified, "
                "since theta and 1/theta give identical autocorrelations"
            ),
            "p": p,
            "q": q,
            "n": int(n),
            "method": "ARMA(%d, %d) by conditional least squares" % (p, q),
        }
    )


def cheatsheet():
    return (
        "hmarma: ARMA(p,q) by conditional least squares with stationarity "
        "and invertibility read off the fitted roots"
    )
