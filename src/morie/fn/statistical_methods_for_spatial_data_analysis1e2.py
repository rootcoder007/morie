# morie.fn -- function file (rootcoder007/morie)
"""Spatial linear model with a parameterised covariance Sigma(theta_s)."""

from . import _array_core as np
from . import _schab_fit as fit

from ._richresult import RichResult
from .spgls import schabenberger_gls_spatial as _gls

__all__ = [
    "spglsm",
    "statistical_methods_for_spatial_data_analysis_chapter_1_equation_2",
]


def spglsm(coords, x, y, nugget=0.0, sill=1.0, range=1.0, model="exponential"):
    r"""GLS for the spatial model with a modelled covariance matrix.

    Equation (1.2), p. 3, is

    .. math::

        \mathbf{Z}(\mathbf{s}) = \mathbf{X}_s\boldsymbol\alpha
        + \boldsymbol\nu, \qquad
        \boldsymbol\nu \sim (\mathbf{0}, \boldsymbol\Sigma(\theta_s)).

    Unlike the longitudinal counterpart (1.3), ``Sigma`` is not
    block-diagonal: there is a single realisation and the whole
    ``n x n`` matrix has to be formed.  Here it is built from an
    isotropic model of Sec. 4.3 evaluated at the interpoint distances,

    ``C(0) = c0 + sigma0^2``, ``C(h) = sigma0^2 R(h)`` for ``h > 0``,

    so the nugget appears only on the diagonal, and the fixed effects are
    then estimated by GLS,
    :math:`\hat{\boldsymbol\alpha} = (\mathbf{X}'\boldsymbol\Sigma^{-1}
    \mathbf{X})^{-1}\mathbf{X}'\boldsymbol\Sigma^{-1}\mathbf{Z}`.

    Parameters
    ----------
    coords : array-like
        Locations, ``(n, d)``.
    x : array-like
        Design matrix ``X_s``, ``(n, p)``.
    y : array-like
        Observed field ``Z(s)``, length ``n``.
    nugget, sill, range : float
        Covariance parameters ``theta_s``.  ``range`` is the practical
        range for the exponential and gaussian families.
    model : str
        ``"exponential"``, ``"gaussian"`` or ``"spherical"``.

    Returns
    -------
    RichResult
        ``beta``, ``vcov``, ``se``, ``residuals``, ``beta_ols``,
        ``se_ols_naive``, ``se_ols_correct``, ``sigma``, ``n``, ``p``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, eq. (1.2), p. 3;
    covariance models Sec. 4.3, pp. 141-152.
    """
    P = np.asarray(coords, dtype=float)
    if P.ndim == 1:
        P = P.reshape((P.size, 1))
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape((X.size, 1))
    z = np.asarray(y, dtype=float).ravel()
    n = int(z.size)
    if int(P.shape[0]) != n or int(X.shape[0]) != n:
        raise ValueError("`coords`, `x` and `y` must have the same number of rows")
    if float(sill) < 0 or float(nugget) < 0:
        raise ValueError("`nugget` and `sill` must be non-negative")
    if float(range) <= 0:
        raise ValueError("`range` must be positive")

    sigma = fit.covariance_matrix(P, float(nugget), float(sill), float(range), model)
    res = _gls(X, z, sigma)
    payload = dict(res)
    payload["sigma"] = sigma
    payload["n"] = n
    payload["p"] = int(X.shape[1])
    return RichResult(
        title="Spatial GLS with modelled Sigma (Schabenberger & Gotway eq. 1.2)",
        summary_lines=[("n", n), ("p", payload["p"]), ("model", model)],
        payload=payload,
    )


statistical_methods_for_spatial_data_analysis_chapter_1_equation_2 = spglsm


def cheatsheet():
    return "spglsm: build Sigma(theta) from an isotropic model, then GLS."
