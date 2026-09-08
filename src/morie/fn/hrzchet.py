# morie.fn -- function file (rootcoder007/morie)
"""Chen (2002) estimator of T in transformation model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_chen_estimator_T"]


def horowitz_chen_estimator_T(x, y, bandwidth=None, beta_hat=None, y0=None,
                              y_grid=None, t_grid=None):
    r"""Chen's (2002) rank estimator of T in the transformation model
    (Horowitz Sec. 6.3.3), equation (6.67):

    .. math:: T_n(y) = \arg\max_{t \in M}\ \frac{1}{n(n-1)}
              \sum_{i}\sum_{j \ne i}
              (d_{iy} - d_{jy_0})\,
              \mathbf 1\{X_i'b_n - X_j'b_n \ge t\},

    with :math:`d_{iy} = \mathbf 1\{Y_i \ge y\}` and
    :math:`d_{jy_0} = \mathbf 1\{Y_j \ge y_0\}`.

    The construction rests on a sign, not a smoothing:
    :math:`E(d_{iy} - d_{jy_0} \mid X_i, X_j) \ge 0` exactly when
    :math:`X_i'\beta - X_j'\beta \ge T(y)`, so the maximiser over t
    of a pairwise sum locates :math:`T(y)`. It is a U-statistic of
    order two -- hence the :math:`n(n-1)` normalisation, and hence
    the cost, which is quadratic in n.

    **This is not faster than Horowitz's estimator.** Both are
    :math:`n^{-1/2}`; Theorem 6.6 gives Chen's the same rate and a
    mean-zero Gaussian-process limit. The book compares them
    directly and reports that neither dominates: Horowitz's tends to
    have smaller mean-square error near the centre of the range of y,
    Chen's further from the centre, and no known estimator is
    asymptotically efficient uniformly over y. A claim that Chen's
    converges faster is wrong, and the returned
    ``faster_than_horowitz`` key says so explicitly.

    ``bandwidth`` is accepted for interface symmetry with
    :mod:`morie.fn.hrzhot` and is NOT used: (6.67) contains no
    kernel. That is one of its attractions.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Covariates.
    y : array-like, shape (n,)
        Response.
    bandwidth : ignored
        Accepted for interface symmetry; (6.67) uses no kernel.
    beta_hat : array-like, optional
        Estimate of beta, rescaled to :math:`|b_1| = 1`. Defaults to
        the first canonical direction.
    y0 : float, optional
        Location-normalisation point; the median of y otherwise.
    y_grid : array-like, optional
        Points at which to return ``T_hat``.
    t_grid : array-like, optional
        The compact interval M searched over (Assumption CT3).

    Returns
    -------
    RichResult
        keys: ``y_grid``, ``T_hat``, ``objective_max``, ``y0``,
        ``beta``, ``t_grid``, ``uses_kernel`` (False),
        ``rate_exponent`` (-1/2), ``faster_than_horowitz`` (False),
        ``n``, ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 6.3.3, eq. (6.67), assumptions
    CT1-CT6 and Theorem 6.6; Chen (2002).
    """
    from ._hrz_transform import SCALE_NOTE, normalize_scale

    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    n, d = X.shape
    if n < 10:
        raise ValueError(f"need at least 10 observations, got {n}.")
    if beta_hat is None:
        b = np.zeros(d)
        b[0] = 1.0
    else:
        b = normalize_scale(beta_hat)
        if b.size != d:
            raise ValueError(
                f"beta_hat has {b.size} entries for {d} covariates.")

    Z = X @ b
    yy0 = float(np.median(yv)) if y0 is None else float(y0)
    yg = np.linspace(np.quantile(yv, 0.1), np.quantile(yv, 0.9), 21) \
        if y_grid is None else np.atleast_1d(np.asarray(y_grid, dtype=float))
    diff = Z[:, None] - Z[None, :]          # X_i'b - X_j'b
    np.fill_diagonal(diff, np.nan)          # j != i
    dj0 = (yv >= yy0).astype(float)[None, :]
    tg = np.linspace(np.nanmin(diff), np.nanmax(diff), 121) \
        if t_grid is None else np.atleast_1d(np.asarray(t_grid, dtype=float))

    T_hat = np.empty(yg.size)
    objmax = np.empty(yg.size)
    denom = n * (n - 1)
    for k, yq in enumerate(yg):
        wgt = (yv >= yq).astype(float)[:, None] - dj0   # d_iy - d_jy0
        vals = np.array([np.nansum(wgt * (diff >= t)) / denom for t in tg])
        j = int(np.argmax(vals))
        T_hat[k] = float(tg[j])
        objmax[k] = float(vals[j])

    return RichResult(payload={
        "y_grid": yg, "T_hat": T_hat, "objective_max": objmax,
        "y0": yy0, "beta": b, "t_grid": tg,
        "uses_kernel": False, "rate_exponent": -0.5,
        "faster_than_horowitz": False,
        "normalisation": SCALE_NOTE, "n": int(n), "d": int(d),
        "method": "Chen (2002) pairwise rank maximisation (6.67); same n^{-1/2} rate as Horowitz"})


def cheatsheet():
    return "hrzchet: a U-statistic with no kernel; SAME rate as Horowitz, neither dominates"
