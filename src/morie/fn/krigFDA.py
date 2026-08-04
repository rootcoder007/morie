"""Universal kriging written in its explicit generalised-least-squares form."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kriging"]


def kriging(coords, values, new_coords, model="exponential", nugget=0.0,
            sill=1.0, range_=1.0, trend_order=1):
    r"""Universal kriging as :math:`Z^*(s_0) = x_0'\hat\beta + \lambda'(Z - X\hat\beta)`.

    Universal kriging is usually solved as one augmented (n+p) system with
    Lagrange multipliers. This module states it the other way, in the
    generalised least squares form the module's own specification asks
    for:

    .. math::
        \hat\beta = (X'\Sigma^{-1}X)^{-1}X'\Sigma^{-1}Z, \qquad
        \lambda = \Sigma^{-1}c_0, \qquad
        Z^*(s_0) = x_0'\hat\beta + \lambda'(Z - X\hat\beta),

    where :math:`\Sigma` is the covariance matrix of the data, :math:`c_0`
    the covariance vector between the data and the target, :math:`X` the
    polynomial trend design matrix and :math:`x_0` its row at the target.
    The trend is estimated by GLS and simple kriging is then applied to
    the GLS residuals. The two formulations are algebraically identical;
    this one has the advantage of exposing :math:`\hat\beta`, which the
    augmented system hides.

    The previous body was a placeholder: it averaged ``coords`` and used
    neither ``values`` nor ``new_coords``.

    Parameters
    ----------
    coords : array-like
        Data locations, shape ``(n, d)``.
    values : array-like
        Observations, length ``n``.
    new_coords : array-like
        Target locations, shape ``(m, d)`` or ``(d,)``.
    model : {'exponential', 'gaussian', 'spherical'}
        Covariance model.
    nugget, sill, range_ : float
        :math:`c_0`, the total sill :math:`c_0 + c_1`, and the range.
    trend_order : {0, 1, 2}
        Polynomial order of the mean function.

    Returns
    -------
    RichResult
        ``estimate`` (length ``m``), ``se``, ``beta`` (the GLS trend
        coefficients), ``residuals`` (:math:`Z - X\hat\beta`), ``weights``
        (:math:`\lambda`, one row per target), ``n``, ``p``, ``method``.

    Notes
    -----
    The prediction variance reported is the universal kriging variance
    from the augmented system, not the simple kriging variance of the
    residual step: the latter would ignore the uncertainty in
    :math:`\hat\beta` and understate the error.

    The conventions for ``model``, ``nugget``, ``sill`` and ``range_``
    are those of :func:`morie.fn.ukrig.universal_kriging`, whose
    augmented-system answer this must reproduce exactly.

    References
    ----------
    Cressie, N. A. C. (1993). *Statistics for Spatial Data*, rev. edn.
    Wiley, sec. 3.4.5 (universal kriging) and 3.4.2.

    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman & Hall/CRC, ch. 5.
    """
    from .ukrig import _cov, _trend

    z = np.asarray(values, dtype=float).ravel()
    s = np.asarray(coords, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    t = np.asarray(new_coords, dtype=float)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    n = int(z.size)
    if s.shape[0] != n:
        raise ValueError("coords rows (%d) must match values (%d)"
                         % (s.shape[0], n))
    if t.shape[1] != s.shape[1]:
        raise ValueError("new_coords dim %d must match coords dim %d"
                         % (t.shape[1], s.shape[1]))
    c0 = float(nugget)
    c1 = float(sill) - c0
    if c1 < 0.0:
        raise ValueError("sill must be >= nugget")
    a = float(range_)

    D = _pdist(s, s)
    Sig = _cov(D, c0, c1, a, model)
    X = _trend(s, trend_order)
    p = int(X.shape[1])

    Si = np.linalg.inv(Sig)
    XtSi = X.T @ Si
    beta = np.linalg.solve(XtSi @ X, XtSi @ z)
    resid = z - X @ beta

    total_var = c0 + c1
    K = np.zeros((n + p, n + p))
    K[:n, :n] = Sig
    K[:n, n:] = X
    K[n:, :n] = X.T

    est = []
    se = []
    wts = []
    for row in t:
        d0 = _pdist(row.reshape(1, -1), s).ravel()
        cv = _cov(d0, c0, c1, a, model)
        x0 = _trend(row.reshape(1, -1), trend_order).ravel()
        lam = Si @ cv
        est.append(float(x0 @ beta + lam @ resid))
        rhs = np.concatenate([cv, x0])
        sol = np.linalg.solve(K, rhs)
        se.append(float(np.sqrt(max(total_var - float(sol @ rhs), 0.0))))
        wts.append([float(v) for v in lam])

    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "beta": [float(v) for v in beta],
            "residuals": [float(v) for v in resid],
            "weights": wts,
            "n": n,
            "p": p,
            "method": "Universal kriging in GLS form, Z* = x0'beta + lambda'(Z - X beta)",
        }
    )


def _pdist(a, b):
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def cheatsheet():
    return "krigFDA: universal kriging as GLS trend plus simple kriging of the residuals"
