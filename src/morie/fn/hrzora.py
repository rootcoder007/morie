# morie.fn -- function file (rootcoder007/morie)
"""Two-step oracle-efficient additive model estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_two_step_oracle"]


def horowitz_two_step_oracle(x, y, bandwidth=None, kappa=None, local_linear=True,
                             grid=None):
    r"""Horowitz-Mammen (2004) two-step oracle-efficient estimator of
    a nonparametric additive model (Horowitz Sec. 3.1.3),
    equation (3.18):

    stage one fits a SERIES approximation by ordinary least squares
    with the additive structure imposed,

    .. math:: \hat\theta_{n\kappa} = \arg\min_\theta \frac1n
              \sum_i \big[Y_i - \Psi_\kappa(X_i)'\theta\big]^2,
              \qquad
              \tilde\mu + \tilde m(x) = \Psi_\kappa(x)'\hat\theta,

    and stage two runs a ONE-DIMENSIONAL nonparametric regression of
    the partialled-out response on the component of interest,

    .. math:: \hat m_{1,K}(x^1) = \frac{\sum_i [Y_i -
              \tilde m_{-1}(X_i^{(-1)})]\,K\!\big((x^1 - X_i^1)/h\big)}
              {\sum_i K\!\big((x^1 - X_i^1)/h\big)},

    with the local-linear variant replacing that Nadaraya-Watson step.

    **The point is what is NOT here.** No stage smooths in more than
    one dimension. The first stage imposes additivity through the
    basis, so it never performs :math:`d`-dimensional nonparametric
    regression; the second stage is a scalar smooth. The estimator is
    therefore :math:`n^{-2/5}` consistent, asymptotically normal and
    ORACLE EFFICIENT for any finite d -- each component is estimated
    as accurately as if every other component were known. That is the
    same rate as estimating a single twice-differentiable function of
    a scalar, so there is no curse of dimensionality, in contrast to
    the marginal integration of :mod:`morie.fn.hrzmir`.

    It is also **not iterative**, which distinguishes it from
    backfitting. Backfitting is defined as the limit of a sequence
    rather than by a formula, which makes its properties hard to
    establish, and Opsomer and Ruppert found the usual version is NOT
    oracle efficient and needs strong restrictions on the
    distribution of X. Two non-iterative stages avoid all of that.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Covariates. Rescaled to :math:`[-1, 1]` per component, the
        support the basis is defined on.
    y : array-like, shape (n,)
        Response.
    bandwidth : float, optional
        Second-stage bandwidth; ``n**(-1/5)`` scaled otherwise, the
        rate that attains :math:`n^{-2/5}`.
    kappa : int, optional
        Series terms per component. Must grow with n; defaults to
        ``ceil(n**(1/5))``, at least 2.
    local_linear : bool, default True
        Use the local-linear second stage, which behaves better at
        the boundary and adapts to non-uniform designs.
    grid : array-like, optional
        Points at which to return each component.

    Returns
    -------
    RichResult
        keys: ``grid``, ``m_hat`` (d by grid), ``mu_hat``,
        ``theta``, ``kappa``, ``bandwidth``, ``oracle_efficient``
        (True), ``iterative`` (False), ``rate_exponent`` (-2/5),
        ``max_smoothing_dimension`` (1),
        ``curse_of_dimensionality`` (False), ``n``, ``d``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 3.1.3, eqs. (3.15)-(3.18) and
    Sec. 3.1.3.2; Horowitz and Mammen (2004).
    """
    from ._horowitz import kernel

    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    n, d = X.shape
    if n < 30:
        raise ValueError(f"need at least 30 observations, got {n}.")
    if d < 2:
        raise ValueError(
            f"an additive model needs at least 2 components, got {d}.")

    # the basis is defined on [-1, 1], so map each component onto it
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Z = 2.0 * (X - lo) / span - 1.0

    kap = int(np.ceil(n ** 0.2)) if kappa is None else int(kappa)
    if kap < 2:
        raise ValueError(f"kappa must be at least 2, got {kap}.")

    def basis(v):
        """psi_k on [-1, 1] satisfying (3.15) and (3.16).

        The full Fourier system minus the constant, alternating
        sin(j pi v), cos(j pi v). Both have integral zero over
        [-1, 1] as (3.15) requires, and both integrate to one against
        themselves and to zero against each other, which is (3.16).

        A cosine-only basis also satisfies (3.15) and (3.16) and is
        the obvious thing to reach for -- but every cos(k pi v) is
        EVEN, so it spans no odd function. It cannot represent an odd
        additive component at all, and the damage shows up in the
        OTHER components: the first stage fails to partial the odd
        one out, so its residual contaminates every second-stage
        regression. Measured on m_1 = sin(pi x), m_2 = x^2 - 1/3, the
        cosine basis recovered m_1 at correlation 0.9997 and m_2 at
        only 0.92, flat in kappa -- adding terms cannot fix an
        incomplete basis.
        """
        cols = []
        for m in range(1, kap + 1):
            k = (m + 1) // 2
            cols.append(np.sin(np.pi * k * v) if m % 2 else
                        np.cos(np.pi * k * v))
        return np.column_stack(cols)

    # (3.17): [1, psi_1(x^1)..psi_k(x^1), .., psi_1(x^d)..psi_k(x^d)]
    Psi = np.column_stack([np.ones(n)] + [basis(Z[:, jj]) for jj in range(d)])
    theta, *_ = np.linalg.lstsq(Psi, yv, rcond=None)
    mu_tilde = float(theta[0])

    def series_component(jj, zv):
        coef = theta[1 + jj * kap: 1 + (jj + 1) * kap]
        return basis(np.asarray(zv, dtype=float)) @ coef

    # one bandwidth PER COMPONENT: they are separate scalar smooths
    # and share nothing but the sample, so scaling them all by the
    # first component's spread is simply wrong when the components
    # differ in shape or spread
    if bandwidth is None:
        hvec = np.array([float(np.std(Z[:, k]) * n ** -0.2) for k in range(d)])
    else:
        hb = np.atleast_1d(np.asarray(bandwidth, dtype=float)).ravel()
        hvec = np.full(d, float(hb[0])) if hb.size == 1 else hb.astype(float)
        if hvec.size != d:
            raise ValueError(
                f"bandwidth must be a scalar or {d} values, got {hvec.size}.")
    if np.any(hvec <= 0):
        raise ValueError(f"bandwidths must be positive, got {hvec.tolist()}.")
    gz = np.linspace(-0.9, 0.9, 41) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))

    m_hat = np.empty((d, gz.size))
    for jj in range(d):
        # partial out every OTHER component using the first-stage fit
        others = sum(series_component(k, Z[:, k]) for k in range(d) if k != jj)
        resid = yv - mu_tilde - others
        zj = Z[:, jj]
        hh = float(hvec[jj])
        for t, v in enumerate(gz):
            w = kernel((v - zj) / hh)
            sw = w.sum()
            if sw <= 0:
                m_hat[jj, t] = np.nan
                continue
            if local_linear:
                dx = zj - v
                s0 = sw
                s1 = float(w @ dx)
                s2 = float(w @ dx**2)
                t0 = float(w @ resid)
                t1 = float(w @ (resid * dx))
                det = s0 * s2 - s1 * s1
                m_hat[jj, t] = (s2 * t0 - s1 * t1) / det if det != 0 \
                    else t0 / s0
            else:
                m_hat[jj, t] = float(w @ resid) / sw
        # impose the location normalisation on the fitted component
        m_hat[jj] -= np.nanmean(m_hat[jj])

    return RichResult(payload={
        "grid": gz, "m_hat": m_hat, "mu_hat": mu_tilde, "theta": theta,
        "kappa": kap, "bandwidth": hvec if d > 1 else float(hvec[0]),
        "oracle_efficient": True, "iterative": False,
        "rate_exponent": -0.4, "max_smoothing_dimension": 1,
        "curse_of_dimensionality": False,
        "n": int(n), "d": int(d),
        "method": "Horowitz-Mammen two-step (3.18); series first, scalar smooth second, no d-dimensional step"})


def cheatsheet():
    return "hrzora: no stage smooths in more than ONE dimension -- oracle efficient, non-iterative, n^{-2/5}"
