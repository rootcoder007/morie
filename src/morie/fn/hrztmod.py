# morie.fn -- function file (rootcoder007/morie)
"""Transformation model: T(Y) = X'beta + U."""

from . import _array_core as np
from . import _horowitz as HZ
from . import _hrz3 as H
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["horowitz_transformation_model"]


def horowitz_transformation_model(x, y, ny=21, nz=21, bandwidth=None):
    r"""Transformation model with nonparametric T, by Horowitz's (1996)
    estimator.

    Horowitz (2009), Section 6.3.1, pages 216-218.  The model is
    :math:`T(Y) = X'\beta + U` with :math:`T` an unknown strictly
    increasing function and :math:`U` independent of :math:`X` with
    unknown CDF :math:`F`.

    With :math:`Z = X'\beta` and :math:`G(\cdot|z)` the CDF of
    :math:`Y` given :math:`Z = z`, the model implies
    :math:`G(y|z) = F[T(y) - z]`, hence
    :math:`G_y = T'F'` and :math:`G_z = -F'`, so
    :math:`T'(y) = -G_y(y|z)/G_z(y|z)` and

    .. math:: T(y) = -\int_{y_0}^{y}\frac{G_y(v|z)}{G_z(v|z)}\,dv
                                                              \quad (6.57)

    Averaging over :math:`z` against a weight :math:`w` supported on a
    compact :math:`S_w` with :math:`\int_{S_w} w = 1` (6.58) gives

    .. math:: T(y) = -\int_{y_0}^{y}\!\!\int_{S_w} w(z)
              \frac{G_y(v|z)}{G_z(v|z)}\,dz\,dv                \quad (6.59)

    and the estimator replaces :math:`G_y, G_z` by kernel estimators
    (6.61)-(6.62):

    .. math:: T_n(y) = -\int_{y_0}^{y}\!\!\int_{S_w} w(z)
              \frac{G_{ny}(v|z)}{G_{nz}(v|z)}\,dz\,dv          \quad (6.60)

    The leading minus signs in (6.57), (6.59) and (6.60) were read off
    a rendered image of page 217, not from an extracted text layer;
    :math:`G_z < 0` so the sign is what makes :math:`T` increasing, and
    dropping it silently inverts the estimate.

    Averaging over :math:`z` is not cosmetic.  The pointwise ratio
    :math:`G_{ny}/G_{nz}` converges more slowly than
    :math:`n^{-1/2}`; integrating over :math:`z` and :math:`v` creates
    the averaging effect that restores the :math:`n^{-1/2}` rate.
    That is why the estimator is based on (6.59) and not on (6.57).

    :math:`T` is estimated only on a compact interval
    :math:`[y_2, y_1]` strictly inside the support of :math:`Y`
    (p. 219): :math:`T` may be unbounded at the boundary and
    :math:`G_z` is likely to vanish there.  The default takes the
    10th to 90th percentiles, and :math:`y_0` is the central grid
    point, so :math:`T_n(y_0) = 0` holds exactly as required by
    HT5(e).

    :math:`\beta` is estimated by the density-weighted average
    derivative of Section 2.6.1 with the scale normalisation
    :math:`|\beta_1| = 1` (HT2(a)); the text notes that (6.1) with
    nonparametric :math:`T` and :math:`F` "is a semiparametric
    single-index model", so any Chapter 2 estimator is admissible.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, d)
        Covariates.  The first column carries the scale normalisation.
    y : array-like, shape (n,)
        Response.
    ny : int, default 21
        Grid points on [y_2, y_1]; forced odd so ``y_0`` is a grid point.
    nz : int, default 21
        Quadrature points on the weight support ``S_w``.
    bandwidth : float, optional
        Common bandwidth; default Silverman's rule per variable.

    Returns
    -------
    RichResult
        keys: ``T_hat``, ``beta_hat``, ``y_grid``, ``y0``, ``y2``,
        ``y1``, ``index``, ``i0``, ``monotone``, ``bandwidth_y``,
        ``bandwidth_z``, ``n``, ``d``, ``method``.

    References
    ----------
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
    in Econometrics*. Springer, Sec. 6.3.1, eqs. (6.57)-(6.62),
    pp. 216-218.
    Horowitz, J. L. (1996). Semiparametric estimation of a regression
    model with an unknown transformation of the dependent variable.
    *Econometrica* 64(1), 103-137.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = int(y.size)
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != n and X.shape[1] == n:
        X = X.T
    if X.shape[0] != n:
        raise ValueError(f"x must have {n} rows, got shape {X.shape}.")
    d = int(X.shape[1])
    if n < 10:
        raise ValueError(f"need at least 10 observations, got {n}.")
    ny = int(ny)
    nz = int(nz)
    if ny < 3 or nz < 3:
        raise ValueError(f"ny and nz must be >= 3, got {ny} and {nz}.")
    if ny % 2 == 0:
        ny += 1

    yl = [float(t) for t in y]
    hy = float(bandwidth) if bandwidth is not None else HZ.silverman_bw(y)

    # beta by density-weighted average derivative, |beta_1| = 1.
    hb = float(bandwidth) if bandwidth is not None else HZ.silverman_bw(
        [float(X[i][0]) for i in range(n)])
    beta = H.index_dir(X, y, hb)
    Z = [0.0] * n
    for i in range(n):
        s = 0.0
        for k in range(d):
            s += float(X[i][k]) * float(beta[k])
        Z[i] = s
    hz = float(bandwidth) if bandwidth is not None else HZ.silverman_bw(Z)

    y2 = core.quantile7(yl, 0.10)
    y1 = core.quantile7(yl, 0.90)
    if not (y1 > y2):
        raise ValueError("Y has no spread between its 10th and 90th percentiles.")
    za = core.quantile7(Z, 0.25)
    zb = core.quantile7(Z, 0.75)
    if not (zb > za):
        raise ValueError("the index has no spread on the weight support S_w.")

    dv = (y1 - y2) / (ny - 1)
    ygrid = [y2 + k * dv for k in range(ny)]
    ygrid[0] = y2
    ygrid[ny - 1] = y1
    i0 = (ny - 1) // 2
    y0 = ygrid[i0]
    dz = (zb - za) / (nz - 1)
    zgrid = [za + k * dz for k in range(nz)]
    wz = [dz] * nz
    wz[0] = dz / 2.0
    wz[nz - 1] = dz / 2.0
    wt = 1.0 / (zb - za)          # w(z) uniform on S_w, satisfies (6.58)

    inner = [0.0] * ny
    for k in range(ny):
        v = ygrid[k]
        KY = [np.exp(-0.5 * ((yl[i] - v) / hy) ** 2) / H.SQRT2PI
              for i in range(n)]
        acc = 0.0
        for q in range(nz):
            zq = zgrid[q]
            A = B = Az = Bz = num = 0.0
            for i in range(n):
                u = (Z[i] - zq) / hz
                kk = np.exp(-0.5 * u * u) / H.SQRT2PI
                dk = (u / hz) * kk            # d/dz K((Z_i - z)/h)
                ind = 1.0 if yl[i] <= v else 0.0
                A += ind * kk
                B += kk
                Az += ind * dk
                Bz += dk
                num += KY[i] * kk
            if B <= 1e-300:
                continue
            Gnz = (Az * B - A * Bz) / (B * B)
            Gny = num / (hy * B)
            if abs(Gnz) < 1e-300:
                continue
            acc += wz[q] * wt * (Gny / Gnz)
        inner[k] = acc

    T = [0.0] * ny
    for k in range(i0 + 1, ny):
        T[k] = T[k - 1] - 0.5 * dv * (inner[k - 1] + inner[k])
    for k in range(i0 - 1, -1, -1):
        T[k] = T[k + 1] + 0.5 * dv * (inner[k] + inner[k + 1])

    monotone = True
    for k in range(1, ny):
        if T[k] < T[k - 1] - 1e-12:
            monotone = False

    return RichResult(payload={
        "T_hat": T,
        "beta_hat": [float(t) for t in beta],
        "y_grid": ygrid,
        "y0": y0,
        "y2": y2,
        "y1": y1,
        "index": Z,
        "i0": i0,
        "monotone": monotone,
        "bandwidth_y": hy,
        "bandwidth_z": hz,
        "n": n,
        "d": d,
        "method": "Horowitz (2009) eq. (6.60), Horowitz (1996) estimator of T",
    })


def cheatsheet():
    return "hrztmod: (6.60) T from averaged Gny/Gnz; the leading minus makes T increasing"
