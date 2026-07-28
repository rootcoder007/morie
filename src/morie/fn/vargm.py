# morie.fn -- function file (rootcoder007/morie)
"""Empirical (Matheron) variogram with a fitted model."""

import numpy as np

from ._richresult import RichResult
from ._schaben import MODELS, fit_variogram_wls, matheron, variogram_model

__all__ = ["empirical_variogram"]


def empirical_variogram(coords, z, lags=None, cutoff=None, model=None,
                        weights="cressie"):
    r"""Empirical semivariogram, optionally with a model fitted to it.

    .. math::
       \gamma(h) = \frac{1}{2|N(h)|}\sum_{(i,j)\in N(h)} (z_i - z_j)^2

    When ``model`` is given, the model is fitted by weighted least
    squares with Cressie's weights (equation 4.34),

    .. math::
       \sum_m \frac{|N(h_m)|}{2\gamma(h_m,\theta)^2}
              \{\hat\gamma(h_m) - \gamma(h_m,\theta)\}^2,

    iteratively re-weighted because the weights themselves depend on
    the parameters. ``weights='ols'`` drops them.

    The book's own assessment of this choice is worth repeating: the
    weighted criterion is a POOR approximation to the generalised one,
    because the off-diagonal covariances between empirical
    semivariogram values are appreciable and neither OLS nor WLS
    accounts for them. Zimmerman and Zimmerman (1991) found the two
    perform about equally; the efficiency that is actually being lost
    is lost to the correlations, not to the weighting.

    Parameters
    ----------
    coords : array-like, shape (n, d)
    z : array-like, shape (n,)
    lags : int or sequence, optional
        Lag-class count or explicit edges.
    cutoff : float, optional
        Largest separation used; half the maximum by default.
    model : {'exponential', 'spherical', 'gaussian', 'linear'}, optional
        Fit this model to the empirical values.
    weights : {'cressie', 'ols'}
        Weighting for the fit.

    Returns
    -------
    RichResult
        ``lag``, ``gamma``, ``n_pairs``, ``variance``, and when a model
        was requested ``nugget``, ``psill``, ``range``, ``sill``,
        ``fitted``, ``relative_nugget``.

    References
    ----------
    Schabenberger and Gotway (2005), sections 4.4.1 and 4.5.1,
    equations (4.24), (4.33) and (4.34), pp. 153-166.
    Matheron (1962). Cressie (1985).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> co = rng.uniform(0, 10, size=(120, 2))
    >>> z = co[:, 0] * 0.5 + rng.normal(scale=0.4, size=120)
    >>> out = empirical_variogram(co, z, lags=8)
    >>> bool(out["gamma"][-1] > out["gamma"][0])
    True
    """
    lag, gam, npair, var = matheron(coords, z, lags, cutoff)
    payload = {
        "estimate": gam,
        "gamma": gam,
        "lag": lag,
        "n_pairs": npair,
        "variance": var,
        "n": int(np.asarray(z).size),
        "method": "Empirical (Matheron) semivariogram",
    }
    if model is not None:
        if model not in MODELS:
            raise ValueError(
                "model must be one of %s, got %r." % (MODELS, model)
            )
        fit = fit_variogram_wls(lag, gam, npair, model, weights)
        rel = fit["nugget"] / fit["sill"] if fit["sill"] > 0 else np.nan
        payload.update({
            "nugget": fit["nugget"],
            "psill": fit["psill"],
            "range": fit["range"],
            "sill": fit["sill"],
            "model": model,
            "weights": weights,
            "fitted": variogram_model(lag, model, fit["nugget"],
                                      fit["psill"], fit["range"]),
            "objective": fit["objective"],
            "iterations": fit["iterations"],
            "relative_nugget": float(rel),
            "range_note": (
                "the exponential and Gaussian models use the PRACTICAL "
                "range, the distance at which correlation has decayed to "
                "0.05; a scale-parameter convention would differ by a "
                "factor of 3"
            ),
            "weights_note": (
                "Cressie's weights (4.34) approximate the diagonal of the "
                "true covariance only; the off-diagonal correlations among "
                "empirical semivariogram values are appreciable and are what "
                "actually costs efficiency"
            ),
            "method": "Empirical semivariogram with a %s model fitted by %s"
                      % (model, "WLS" if weights == "cressie" else "OLS"),
        })
    return RichResult(payload=payload)


def cheatsheet():
    return (
        "vargm: empirical semivariogram, optionally with a nugget/sill/range "
        "model fitted by Cressie-weighted least squares"
    )
