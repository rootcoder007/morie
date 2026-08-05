# morie.fn -- function file (rootcoder007/morie)
"""Partially linear quantile model."""

from . import _array_core as np
from . import _horowitz as HZ
from . import _hrz3 as H

from ._richresult import RichResult

__all__ = ["horowitz_plr_quantile"]


def _asmat(x, n):
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != n and X.shape[1] == n:
        X = X.T
    if X.shape[0] != n:
        raise ValueError(f"x must have {n} rows, got shape {X.shape}.")
    return X


def horowitz_plr_quantile(x, y, z, bandwidth=None, tau=0.5, niter=12):
    r"""Partially linear model of a conditional quantile.

    Horowitz (2009), Section 3.6.3, pages 90-91.  The model is

    .. math:: Q_\tau(Y|X, Z) = X'\beta + g(Z),

    equivalently :math:`Y = X'\beta + g(Z) + U` with
    :math:`P(U \le 0|X=x, Z=z) = \tau`.

    The docstring inherited from the stub described a "Robinson
    approach".  Page 90 says in terms why that is not available:
    Robinson's differencing works for (3.2a) "because the mean of a
    sum of random variables is the sum of the individual means.  The
    quantile of the sum of random variables is not the sum of the
    individual quantiles.  Consequently, differencing cannot be used
    to eliminate g from (3.2b)".  The estimator implemented here is
    therefore the one the section actually gives, due to Chen, Linton
    and Van Keilegom (2003), based on the moment condition

    .. math:: E\,X\{\tau - I[Y - X'\beta - g(Z) \le 0]\} = 0  \quad (3.38)

    with :math:`g` replaced by a nonparametric estimator that is
    re-computed at every candidate :math:`b`:

    .. math:: \hat b = \arg\min_b \Big\|\frac{1}{n}\sum_i X_i
              \{\tau - I[Y_i - X_i'b - \hat g(Z_i, b) \le 0]\}\Big\|^2

    where :math:`\hat g(z, b)` is the kernel-weighted
    :math:`\tau`-quantile of :math:`Y - X'b` given :math:`Z = z`.  The
    re-estimation of :math:`g` inside the loop is the point: with
    :math:`g` held fixed at a first-pass value the criterion is not
    the one (3.38) identifies.

    The criterion is a STEP function of :math:`b`, so a
    gradient-based optimiser is inapplicable.  A fixed-schedule
    coordinate search is used, with no tolerance-based exit and no
    random restart, so both language arms take the same path.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, p)
        Covariates entering linearly.
    y : array-like, shape (n,)
        Response.
    z : array-like, shape (n,)
        Covariate entering through the nonparametric ``g``.
    bandwidth : float, optional
        Bandwidth for ``g``.  Default: Silverman's rule on ``z``.
    tau : float, default 0.5
        Quantile level in (0, 1).
    niter : int, default 12
        Coordinate-search sweeps.

    Returns
    -------
    RichResult
        keys: ``beta_tau``, ``g_tau_hat`` (at the observed Z, input
        order), ``criterion``, ``tau``, ``bandwidth``, ``n``, ``p``,
        ``method``.

    References
    ----------
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
    in Econometrics*. Springer, Sec. 3.6.3, eq. (3.38), pp. 90-91.
    Chen, X., Linton, O. & Van Keilegom, I. (2003). Estimation of
    semiparametric models when the criterion function is not smooth.
    *Econometrica* 71(5), 1591-1608.
    """
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    n = int(y.size)
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")
    if z.size != n:
        raise ValueError(f"y has {n} points but z has {z.size}.")
    X = _asmat(x, n)
    p = int(X.shape[1])
    tau = float(tau)
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must lie strictly in (0, 1), got {tau}.")
    hz = float(bandwidth) if bandwidth is not None else HZ.silverman_bw(z)
    if hz <= 0:
        raise ValueError(f"bandwidth must be positive, got {hz}.")

    # Kernel weights for g are fixed: they depend on Z only, not on b.
    Wz = H.kmat(z, z, hz)

    def resid(b):
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            for k in range(p):
                s += float(X[i][k]) * b[k]
            out[i] = float(y[i]) - s
        return out

    def ghat(r):
        return [H.wquant(r, [float(Wz[i][j]) for j in range(n)], tau)
                for i in range(n)]

    def crit(b):
        r = resid(b)
        g = ghat(r)
        acc = [0.0] * p
        for i in range(n):
            ind = 1.0 if (r[i] - g[i]) <= 0.0 else 0.0
            for k in range(p):
                acc[k] += float(X[i][k]) * (tau - ind)
        s = 0.0
        for k in range(p):
            s += (acc[k] / n) ** 2
        return s

    b0 = np.linalg.lstsq(np.asarray(X, dtype=float),
                         np.asarray(y, dtype=float), rcond=None)[0]
    b_hat, value = HZ.coord_min(crit, [float(t) for t in b0],
                                niter=int(niter), delta=1.0, shrink=0.5,
                                steps=3)
    g_hat = ghat(resid(b_hat))

    return RichResult(payload={
        "beta_tau": [float(t) for t in b_hat],
        "g_tau_hat": g_hat,
        "criterion": float(value),
        "tau": tau,
        "bandwidth": hz,
        "n": n,
        "p": p,
        "method": "Horowitz (2009) eq. (3.38), Chen-Linton-Van Keilegom",
    })


def cheatsheet():
    return "hrzplrq: (3.38); differencing does NOT work for quantiles (p.90)"
