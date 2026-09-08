# morie.fn -- function file (rootcoder007/morie)
"""Tikhonov regularization for NPIV when T is unknown."""

from . import _array_core as np
from . import _hrz3 as H

from ._richresult import RichResult

__all__ = ["horowitz_tikhonov_unknown_T"]


def horowitz_tikhonov_unknown_T(x, y, w, bandwidth=None, alpha=1e-3,
                                grid=25):
    r"""Tikhonov regularisation for nonparametric IV when T is unknown.

    Horowitz (2009), *Semiparametric and Nonparametric Methods in
    Econometrics*, Section 5.4.1, pages 171-172.  For the model
    :math:`Y = g(X) + U` with :math:`E(U|W=w) = 0`, both the operator
    :math:`T` and the joint density :math:`f_{XW}` are unknown and are
    replaced by Nadaraya-Watson kernel estimators.  With

    .. math:: \hat r(z) = n^{-1}\sum_{i=1}^{n}
              Y_i\,\hat f^{(-i)}_{XW}(z, W_i)                  \quad (5.71)

    and :math:`\hat t(x,z) = \int_0^1 \hat f_{XW}(x,w)
    \hat f_{XW}(z,w)\,dw` defining :math:`\hat T`, the estimator is

    .. math:: \hat g = (\hat T + a_n)^{-1}\hat r.               \quad (5.72)

    The leave-one-out density :math:`\hat f^{(-i)}_{XW}` is used in
    :math:`\hat r` "to avoid biases that arise if the estimator of
    ``f_XW`` is not statistically independent of its argument"
    (p. 171).  Dropping the leave-one-out correction is the single
    easiest way to get a plausible-looking but biased answer here.

    Because :math:`\hat t` is already the Gram kernel of
    :math:`\hat f_{XW}`, :math:`\hat T` is self-adjoint and positive
    semi-definite, so (5.72) coincides with the normal-equation form
    :math:`(\hat T^{*}\hat T + \alpha I)^{-1}\hat T^{*}\hat m` quoted
    in the older literature; no separate :math:`\hat T^{*}\hat T` is
    formed.

    Two printing errors in the source were corrected here, both
    confirmed against a rendered image of page 171:

    * the displayed :math:`\hat f_{XW}` carries the factor
      :math:`1/(nh_n)` while summing a PRODUCT of two kernels; a
      bivariate kernel density needs :math:`1/(nh_n^2)`.  The constant
      does not cancel, because :math:`\hat r` is linear and
      :math:`\hat T` quadratic in :math:`\hat f_{XW}`.  The returned
      ``fxw`` integrates to one over the unit square, which the
      printed constant would not.
    * the displayed :math:`\hat t(x,z)` integrates ``dz``; it must be
      ``dw``, matching the population definition (5.43) on page 157.

    Parameters
    ----------
    x : array-like, shape (n,)
        Endogenous regressor.
    y : array-like, shape (n,)
        Response.
    w : array-like, shape (n,)
        Instrument.
    bandwidth : float, optional
        Kernel bandwidth :math:`h_n` on the [0, 1] scale.  Default
        ``1.06 n^(-1/6)/sqrt(12)``: after the mid-rank map the marginals
        are exactly uniform, so Silverman's constant fixes the scale and
        not merely the rate.
    alpha : float, default 1e-3
        Regularisation constant :math:`a_n` in (5.72).  Must be
        positive: at ``alpha = 0`` the problem is ill-posed and the
        solve is meaningless, which is the whole point of Section 5.4.
    grid : int, default 25
        Number of quadrature points on [0, 1].

    Returns
    -------
    RichResult
        keys: ``g_hat`` (on ``grid_points``), ``grid_points``,
        ``r_hat``, ``fxw``, ``raw_mass``, ``alpha``, ``bandwidth``, ``n``, ``m``,
        ``method``.

    References
    ----------
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
    in Econometrics*. Springer, Sec. 5.4.1, eqs. (5.71)-(5.72),
    pp. 171-172.
    Hall, P. & Horowitz, J. L. (2005). Nonparametric methods for
    inference in the presence of instrumental variables.
    *Annals of Statistics* 33(6), 2904-2929.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = int(x.size)
    if y.size != n or w.size != n:
        raise ValueError(
            f"x, y, w must have the same length; got {n}, {y.size}, {w.size}.")
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}.")
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError(
            f"alpha must be positive; the problem is ill-posed at 0, got {alpha}.")
    h = float(bandwidth) if bandwidth is not None else H.bw01(n)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")

    u = H.u01(x)
    v = H.u01(w)
    z, wq = H.grid_w(grid)
    m = int(z.size)

    KX = H.kmat(z, u, h)
    KWo = H.kmat(v, v, h)

    # f_hat_XW on the grid, mass-corrected at the [0,1]^2 boundary.
    fxw, raw_mass = H.fxw_grid(u, v, z, wq, h)

    # Leave-one-out density at (z_k, W_i), then r_hat by eq. (5.71).
    r_hat = [0.0] * m
    for k in range(m):
        acc = 0.0
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j == i:
                    continue
                s += float(KX[k][j]) * float(KWo[i][j])
            acc += float(y[i]) * s / ((n - 1) * h * h * raw_mass)
        r_hat[k] = acc / n

    # t_hat(x, z) = int f_hat(x, w) f_hat(z, w) dw, eq. (5.43) form.
    that = [[0.0] * m for _ in range(m)]
    for k in range(m):
        for l in range(m):
            s = 0.0
            for q in range(m):
                s += float(wq[q]) * fxw[k][q] * fxw[l][q]
            that[k][l] = s

    # (T_hat h)(z_l) = sum_k wq_k t_hat(z_k, z_l) h_k; solve (T + a I) g = r.
    A = [[that[k][l] * float(wq[k]) for k in range(m)] for l in range(m)]
    for k in range(m):
        A[k][k] += alpha
    g_hat = np.linalg.solve(np.asarray(A, dtype=float),
                            np.asarray(r_hat, dtype=float))

    return RichResult(payload={
        "g_hat": [float(t) for t in g_hat],
        "grid_points": [float(t) for t in z],
        "r_hat": [float(t) for t in r_hat],
        "fxw": fxw,
        "alpha": alpha,
        "raw_mass": raw_mass,
        "bandwidth": h,
        "n": n,
        "m": m,
        "method": "Horowitz (2009) eq. (5.72), g = (That + a_n)^{-1} rhat",
    })


def cheatsheet():
    return "hrztiku: (5.72) g = (That + a_n)^-1 rhat; leave-one-out rhat is not optional"
