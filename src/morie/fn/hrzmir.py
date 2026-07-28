# morie.fn -- function file (rootcoder007/morie)
"""Marginal integration estimator for additive model components."""

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_marginal_integration"]


def horowitz_marginal_integration(x, y, bandwidth=None, j=0, grid=None):
    r"""Marginal integration for a nonparametric additive model
    (Horowitz Sec. 3.1.1), equations (3.5)-(3.9):

    .. math:: E(Y|X = x) = \mu + m_1(x^1) + \dots + m_d(x^d),

    identified by the location normalisation
    :math:`E[m_j(X^j)] = 0` (3.6), under which :math:`\mu = E(Y)`
    (3.7) and

    .. math:: m_1(x^1) = \int E(Y|X = x)\,p_{-1}(x^{-1})\,dx^{-1}
              - \mu.

    Replacing the conditional mean with the kernel estimator (3.9)
    and the outer integral with a sample average gives

    .. math:: \hat m_1(x^1) = \frac1n\sum_i
              \hat g\big(x^1, X_i^{(-1)}\big) - \hat\mu.

    The idea is exact and the implementation is direct: hold the
    component of interest fixed, average the fitted surface over the
    OTHERS at their observed values, subtract the mean. The
    normalisation is what makes that meaningful -- without
    :math:`E[m_j] = 0` each :math:`m_j` could absorb a constant and
    :math:`\mu` could shed one, so nothing would be identified.

    **This estimator carries the curse of dimensionality**, and it is
    the reason the chapter goes on to develop others. :math:`\hat g`
    in (3.9) smooths over the full :math:`d`-dimensional covariate
    with a :math:`(d-1)`-dimensional kernel :math:`K_2`, so Theorem
    3.1 needs the components to be :math:`q` times differentiable
    for some :math:`q > d - 1`: the requirement grows with d. The
    book's own summary is that marginal integration is
    "conceptually simple but can be hard to compute and does not
    work well when d is large". ``smoothness_required`` returns that
    :math:`q` so the cost is a number rather than a caveat, and it
    is exactly what the two-step estimator of
    :mod:`morie.fn.hrzora` avoids.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Covariates, continuously distributed.
    y : array-like, shape (n,)
        Response.
    bandwidth : float or pair, optional
        ``h_1`` for the component of interest and ``h_2`` for the
        integrated-out directions; Silverman's rule otherwise. The
        two are separate in (3.9) and are kept separate here.
    j : int, default 0
        Which component to estimate.
    grid : array-like, optional
        Points at which to return the component.

    Returns
    -------
    RichResult
        keys: ``grid``, ``m_hat``, ``mu_hat``, ``component``,
        ``h1``, ``h2``, ``normalisation``, ``mean_of_m_hat``
        (near zero by construction), ``smoothness_required``,
        ``curse_of_dimensionality`` (True), ``n``, ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 3.1.1, eqs. (3.5)-(3.9) and
    Theorem 3.1; Linton and Nielsen (1995), Linton and Hardle (1996).
    """
    from ._horowitz import kernel, silverman_bw

    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    n, d = X.shape
    if n < 20:
        raise ValueError(f"need at least 20 observations, got {n}.")
    if d < 2:
        raise ValueError(
            f"an additive model needs at least 2 components, got {d}.")
    jj = int(j)
    if not 0 <= jj < d:
        raise ValueError(f"j must lie in 0..{d - 1}, got {jj}.")

    hb = np.atleast_1d(np.asarray(bandwidth, dtype=float)).ravel() \
        if bandwidth is not None else None
    xj = X[:, jj]
    rest = np.delete(X, jj, axis=1)
    if hb is None:
        h1 = float(silverman_bw(xj))
        # the integrated-out directions are smoothed jointly, so their
        # bandwidth is inflated by the dimension they span
        h2 = float(np.mean([silverman_bw(rest[:, k]) for k in range(d - 1)])
                   * n ** (1.0 / 5.0 - 1.0 / (4.0 + d)))
    elif hb.size == 1:
        h1 = h2 = float(hb[0])
    else:
        h1, h2 = float(hb[0]), float(hb[1])
    if h1 <= 0 or h2 <= 0:
        raise ValueError(f"bandwidths must be positive, got {(h1, h2)}.")

    g = np.linspace(np.quantile(xj, 0.05), np.quantile(xj, 0.95), 41) \
        if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    mu = float(yv.mean())                                   # (3.7)

    # (3.9): K_1 on the held component, K_2 (a product kernel) on the
    # rest; then average over the observed X^{(-1)} and subtract mu
    K2 = np.prod(
        kernel((rest[:, None, :] - rest[None, :, :]) / h2), axis=2)   # (n, n)
    m_hat = np.empty(g.size)
    for t, v in enumerate(g):
        k1 = kernel((v - xj) / h1)                          # (n,)
        num = K2 * (k1 * yv)[None, :]
        den = K2 * k1[None, :]
        ds = den.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ghat = np.where(ds > 0, num.sum(axis=1) / np.maximum(ds, 1e-300),
                            np.nan)
        m_hat[t] = float(np.nanmean(ghat)) - mu
    q_req = d  # Theorem 3.1(b): q > d - 1, so the smallest integer is d

    return RichResult(payload={
        "grid": g, "m_hat": m_hat, "mu_hat": mu, "component": jj,
        "h1": h1, "h2": h2,
        "normalisation": "E[m_j(X^j)] = 0 for every j, so mu = E(Y)",
        "mean_of_m_hat": float(np.nanmean(m_hat)),
        "smoothness_required": int(q_req),
        "curse_of_dimensionality": True,
        "n": int(n), "d": int(d),
        "method": "Marginal integration (3.8)/(3.9); simple, but K_2 is (d-1)-dimensional"})


def cheatsheet():
    return "hrzmir: needs q > d-1 derivatives -- the curse is in the SMOOTHNESS, and hrzora avoids it"
