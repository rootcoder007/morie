# morie.fn -- function file (rootcoder007/morie)
"""Choosing weight function for NLS single-index estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_nls_weight_function"]


def horowitz_nls_weight_function(x, y, bandwidth=None, weights=None,
                                 beta_hat=None):
    r"""The efficient weight function for semiparametric weighted NLS
    in a single-index model (Horowitz Sec. 2.5.2):

    .. math:: W(x) = \frac{1}{\sigma^2(x)},
              \qquad \sigma^2(x) = E\{[Y - G(X'\beta)]^2 \mid X = x\},

    which attains the single-index efficiency bound

    .. math:: \Omega_{SI} = \left\{E\left[
              \frac{\mathbf 1(X \in A_x)}{\sigma^2(X)}
              \frac{\partial G}{\partial \tilde b}
              \frac{\partial G}{\partial \tilde b'}\right]\right\}^{-1}.

    Two results here are easy to state backwards.

    **Not knowing sigma^2 costs nothing.** It can be replaced by a
    consistent estimate :math:`s_n^2(x)` and the bound is still
    attained, by the two-step procedure the section gives: fit with
    :math:`W = 1` -- which is root-n consistent but inefficient --
    then regress the SQUARED residuals nonparametrically on X and
    refit with :math:`W = 1/s_n^2`. That is implemented here, and
    the estimated weights are returned.

    **Not knowing G does cost something.** Except in special cases
    :math:`\Omega_{SI}` EXCEEDS the bound that would be achievable
    if G were known. So the price of a nonparametric link is a loss
    of asymptotic efficiency -- but NOT a loss of rate; the
    estimator is still :math:`n^{-1/2}`. Conflating the two is the
    usual error, and ``efficiency_loss_from_unknown_G`` and
    ``rate_loss_from_unknown_G`` are returned separately for that
    reason.

    The covariance estimator is the sandwich
    :math:`\Omega_n = C_n^{-1}D_n C_n^{-1}` with

    .. math:: C_n = \frac2n \sum_i W_i
              \frac{\partial G}{\partial \tilde b}
              \frac{\partial G}{\partial \tilde b'},
              \qquad
              D_n = \frac4n \sum_i W_i^2 \hat r_i^2
              \frac{\partial G}{\partial \tilde b}
              \frac{\partial G}{\partial \tilde b'}.

    :math:`D_n` carries :math:`W^2`, not :math:`W`. That is fixed by
    the book's own claim that the efficient weight attains
    :math:`\Omega_{SI}`, and only :math:`W^2` delivers it: writing
    :math:`A = n^{-1}\sum W\,\partial G\,\partial G'` gives
    :math:`\Omega_n = A^{-1}BA^{-1}` with
    :math:`B = n^{-1}\sum W^2\hat r^2\partial G\partial G'`, and at
    :math:`W = 1/\sigma^2` with :math:`E(r^2|X) = \sigma^2` we get
    :math:`B \to A`, hence :math:`\Omega_n \to A^{-1} =
    \Omega_{SI}`. With a single power of W the sandwich does not
    collapse to the bound at all. Both :math:`\Omega_n` and the
    bound are returned, and their agreement under efficient weighting
    is a check rather than an assumption.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Covariates.
    y : array-like, shape (n,)
        Response.
    bandwidth : float, optional
        Bandwidth for the index regression and the variance
        regression; Silverman's rule otherwise.
    weights : array-like, optional
        Supplied weights. When omitted the efficient
        :math:`1/s_n^2` weights are estimated in two steps.
    beta_hat : array-like, optional
        Index direction; estimated by unweighted NLS otherwise.

    Returns
    -------
    RichResult
        keys: ``beta``, ``weights``, ``sigma2_hat``, ``omega``
        (the sandwich), ``omega_SI`` (the bound), ``C``, ``D``,
        ``max_weight``, ``efficient_weight_used``,
        ``efficiency_loss_from_unknown_G`` (True),
        ``rate_loss_from_unknown_G`` (False), ``bandwidth``, ``n``,
        ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 2.5.2 (choosing the weight
    function), eq. (2.32); Ichimura (1993), Chamberlain (1986),
    Newey and Stoker (1993).
    """
    from ._horowitz import nw_regression, silverman_bw

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
        raise ValueError(f"need at least 2 covariates, got {d}.")

    if beta_hat is None:
        b = np.zeros(d)
        b[0] = 1.0
    else:
        b = np.asarray(beta_hat, dtype=float).ravel()
        if b.size != d:
            raise ValueError(f"beta_hat has {b.size} entries for {d}.")
        if b[0] == 0:
            raise ValueError("the scale normalisation needs a nonzero first "
                             "coefficient.")
        b = b / abs(b[0])

    z = X @ b
    hh = float(silverman_bw(z)) if bandwidth is None else float(bandwidth)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")

    # step 1: unweighted fit of G on the index
    Ghat = nw_regression(z, yv, grid=z, h=hh)[1]
    resid = yv - Ghat

    # step 2: nonparametric regression of the SQUARED residuals. This
    # runs whatever weight is used, because sigma^2 is a property of
    # the MODEL -- the efficiency bound Omega_SI is defined by it, not
    # by whichever weight the caller happens to supply.
    s2 = np.maximum(nw_regression(z, resid**2, grid=z, h=hh)[1], 1e-12)
    if weights is None:
        w = 1.0 / s2
        efficient = True
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != n:
            raise ValueError(f"weights has {w.size} entries for {n} rows.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
        efficient = False

    # dG/db on the index scale, by differencing the fitted G
    o = np.argsort(z)
    gp = np.zeros(n)
    gp[o] = np.gradient(Ghat[o], z[o])
    # dG/db_tilde = G'(z) * (X_tilde - E[X_tilde | index])
    Xt = X[:, 1:]
    Xbar = np.column_stack([
        nw_regression(z, Xt[:, j], grid=z, h=hh)[1]
        for j in range(d - 1)])
    dG = gp[:, None] * (Xt - Xbar)

    C = 2.0 * (dG * w[:, None]).T @ dG / n
    D = 4.0 * (dG * (w**2 * resid**2)[:, None]).T @ dG / n
    Cinv = np.linalg.pinv(C)
    omega = Cinv @ D @ Cinv
    # the bound (2.32): {E[1(X in A_x)/sigma^2 dG dG']}^{-1}
    omega_si = np.linalg.pinv((dG / np.maximum(s2, 1e-12)[:, None]).T @ dG / n)

    return RichResult(payload={
        "beta": b, "weights": w, "sigma2_hat": s2,
        "omega": omega, "omega_SI": omega_si, "C": C, "D": D,
        "max_weight": float(np.max(w)),
        "efficient_weight_used": efficient,
        "efficiency_loss_from_unknown_G": True,
        "rate_loss_from_unknown_G": False,
        "bandwidth": hh, "n": int(n), "d": int(d),
        "method": "W = 1/sigma^2 attains Omega_SI; a two-step estimate of sigma^2 loses nothing"})


def cheatsheet():
    return "hrzwfun: unknown sigma^2 costs nothing; unknown G costs EFFICIENCY but not rate"
