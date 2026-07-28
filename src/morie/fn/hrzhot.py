# morie.fn -- function file (rootcoder007/morie)
"""Horowitz estimators for T and F in fully nonparametric transformation model."""

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_T_F_estimators"]


def horowitz_T_F_estimators(x, y, bandwidth, beta_hat, y0=None,
                            y_grid=None, u_grid=None, y1=None, y2=None):
    r"""Horowitz's (1996) nonparametric estimators of T and F in the
    transformation model :math:`T(Y) = X'\beta + U` (Horowitz
    Sec. 6.3.1), equations (6.60) and (6.66):

    .. math:: T_n(y) = -\int_{y_0}^{y}\!\int_{S_w} w(z)\,
              \frac{G_{ny}(v|z)}{G_{nz}(v|z)}\,dz\,dv,
              \qquad
              F_n(u) = \frac{A_n(u)}{B_n(u)},

    with

    .. math:: A_n(u) = \frac1n\sum_i \mathbf 1\{U_{ni} \le u\}\,
              \mathbf 1\{T_n(y_2) - u < Z_{ni} \le T_n(y_1) - u\},

    and :math:`B_n` the same without the first indicator.

    The derivation is the point. Differentiating :math:`G(y|z) =
    F[T(y) - z]` gives :math:`T'(y) = -G_y(y|z)/G_z(y|z)`, so T is an
    INTEGRAL of a ratio of kernel estimators. Each of those converges
    more slowly than :math:`n^{-1/2}` and their ratio is not
    root-n-consistent for anything -- but integrating over v and z
    averages the noise away, and that is why the estimator is built
    on (6.59) rather than on the pointwise (6.57). The same averaging
    trick makes density-weighted average derivatives root-n in
    Chapter 2.

    F is NOT the empirical distribution function of
    :math:`U_n = T_n(Y) - X'b_n`, and the book is explicit about why:
    T is root-n estimable only over a compact interval
    :math:`[y_2, y_1]` strictly inside the support of Y, since T may
    be unbounded at the boundaries (for :math:`Y` on
    :math:`[0, \infty)` and :math:`T(y) = \log y` it is). Outside
    that window the :math:`U_{ni}` behave like CENSORED observations,
    and (6.66) is what stays consistent under that censoring.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Covariates. No intercept: the location normalisation
        :math:`T(y_0) = 0` already fixes location.
    y : array-like, shape (n,)
        Response.
    bandwidth : float or pair of floats
        ``h_ny`` and ``h_nz``. A scalar is used for both. Assumption
        HT9 wants them at different rates -- see
        :mod:`morie.fn.hrztfap`.
    beta_hat : array-like, shape (d,)
        A root-n-consistent estimate of beta, rescaled here to
        :math:`|b_1| = 1`.
    y0 : float, optional
        Location-normalisation point; the median of y otherwise.
    y_grid : array-like, optional
        Points at which to return ``T_hat``.
    u_grid : array-like, optional
        Points at which to return ``F_hat``.
    y1, y2 : float, optional
        The trimming window :math:`y_2 < y_1` used by (6.66);
        the 10th and 90th percentiles of y otherwise.

    Returns
    -------
    RichResult
        keys: ``y_grid``, ``T_hat``, ``u_grid``, ``F_hat``, ``beta``,
        ``y0``, ``window`` (y2, y1), ``h_ny``, ``h_nz``,
        ``F_is_empirical_cdf`` (False), ``normalisation``, ``n``,
        ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 6.3.1, eqs. (6.57)-(6.66);
    Horowitz (1996).
    """
    from ._hrz_transform import (SCALE_NOTE, kernel_K, kernel_Kz_sixth,
                                 kernel_Kz_sixth_deriv, normalize_scale)

    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    n, d = X.shape
    if n < 20:
        raise ValueError(f"need at least 20 observations, got {n}.")
    b = normalize_scale(beta_hat)
    if b.size != d:
        raise ValueError(f"beta_hat has {b.size} entries for {d} covariates.")

    hb = np.atleast_1d(np.asarray(bandwidth, dtype=float)).ravel()
    h_ny, h_nz = (float(hb[0]), float(hb[0])) if hb.size == 1 else \
        (float(hb[0]), float(hb[1]))
    if h_ny <= 0 or h_nz <= 0:
        raise ValueError(f"bandwidths must be positive, got {(h_ny, h_nz)}.")

    Z = X @ b
    yy0 = float(np.median(yv)) if y0 is None else float(y0)
    yg = np.linspace(np.quantile(yv, 0.05), np.quantile(yv, 0.95), 41) \
        if y_grid is None else np.atleast_1d(np.asarray(y_grid, dtype=float))

    # w is a weight on z with compact support S_w, integrating to 1
    # (6.58). The interquartile range of Z keeps S_w where p_Z is
    # bounded away from zero, which (6.58)(a) requires.
    zlo, zhi = np.quantile(Z, [0.25, 0.75])
    zs = np.linspace(zlo, zhi, 25)
    w = np.full(zs.size, 1.0 / (zhi - zlo)) if zhi > zlo else np.ones(zs.size)

    def _ratio(v, z):
        """G_ny(v|z) / G_nz(v|z) from (6.61)-(6.62).

        G_n(y|z) = N(z)/D(z) with N the I(Y<=y)-weighted kernel sum
        and D = p_nZ, so G_nz = dG_n/dz comes from the quotient rule
        with dN/dz and dD/dz carrying the -1/h_nz from the chain
        rule.
        """
        a = (Z - z) / h_nz
        kz = kernel_Kz_sixth(a)
        kzp = kernel_Kz_sixth_deriv(a)
        ind = (yv <= v).astype(float)
        dd = kz.sum() / (n * h_nz)                     # D = p_nZ(z)
        if dd <= 0:
            return 0.0
        nn_ = float(np.sum(ind * kz)) / (n * h_nz)     # N(z)
        d_dd = -kzp.sum() / (n * h_nz**2)              # dD/dz
        d_nn = -float(np.sum(ind * kzp)) / (n * h_nz**2)
        g_nz = (d_nn * dd - nn_ * d_dd) / dd**2
        if g_nz == 0:
            return 0.0
        g_ny = float(np.sum(kernel_K((yv - v) / h_ny) * kz)) / \
            (n * h_ny * h_nz * dd)
        return g_ny / g_nz

    def _T_on(points):
        """T_n at each point, integrating (6.60) outward from y0.

        The integrand is shared across points, so it is evaluated
        once on a common v-grid and cumulated, rather than
        re-integrated per point.
        """
        pts = np.asarray(points, dtype=float)
        lo = min(float(pts.min()), yy0)
        hi = max(float(pts.max()), yy0)
        vs = np.linspace(lo, hi, 61)
        inner = np.array([
            np.trapezoid([w[k] * _ratio(v, zs[k]) for k in range(zs.size)], zs)
            for v in vs])
        # -integral from y0 to y, so cumulate then re-base at y0
        cum = np.concatenate([[0.0], np.cumsum(
            np.diff(vs) * (inner[:-1] + inner[1:]) / 2.0)])
        base = np.interp(yy0, vs, cum)
        return -(np.interp(pts, vs, cum) - base), vs, cum

    T_hat, _vs, _cum = _T_on(yg)
    _base = np.interp(yy0, _vs, _cum)

    def _T(q):
        return float(-(np.interp(q, _vs, _cum) - _base))

    # (6.66): F_n = A_n / B_n over the trimming window
    q2 = float(np.quantile(yv, 0.10)) if y2 is None else float(y2)
    q1 = float(np.quantile(yv, 0.90)) if y1 is None else float(y1)
    if q2 >= q1:
        raise ValueError(f"the trimming window needs y2 < y1, got {(q2, q1)}.")
    T_y2, T_y1 = _T(q2), _T(q1)
    # T_n is a function of y alone, so it is interpolated off the
    # same integration grid rather than re-integrated per observation
    Uni = -(np.interp(yv, _vs, _cum) - _base) - Z
    ug = np.linspace(np.quantile(Uni, 0.1), np.quantile(Uni, 0.9), 41) \
        if u_grid is None else np.atleast_1d(np.asarray(u_grid, dtype=float))
    F_hat = np.empty(ug.size)
    for j, u in enumerate(ug):
        inwin = (T_y2 - u < Z) & (Z <= T_y1 - u)
        Bn = float(np.mean(inwin))
        An = float(np.mean((Uni <= u) & inwin))
        F_hat[j] = An / Bn if Bn > 0 else np.nan

    return RichResult(payload={
        "y_grid": yg, "T_hat": T_hat, "u_grid": ug, "F_hat": F_hat,
        "beta": b, "y0": yy0, "window": (q2, q1),
        "h_ny": h_ny, "h_nz": h_nz,
        "F_is_empirical_cdf": False,
        "normalisation": SCALE_NOTE, "n": int(n), "d": int(d),
        "method": "Horowitz (1996) T_n (6.60) and F_n (6.66); F is not the EDF of U_n"})


def cheatsheet():
    return "hrzhot: integrating the ratio is what buys root-n; F_n is NOT the EDF of U_n"
