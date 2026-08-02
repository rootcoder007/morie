# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ordinary least squares fitting of a semivariogram model."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_fit import as_empirical_variogram, fit_semivariogram
from ._schab_vario import semivariogram

__all__ = ["schabenberger_ols_variogram"]


def schabenberger_ols_variogram(empirical_variogram, variogram_model="exponential"):
    """Fit a semivariogram model by ordinary least squares.

    Minimises

        sum_m {gamma_hat(h_m) - gamma(h_m; theta)}^2

    which is the generalized sum of squares (4.31) under the simplification
    R = phi * I named in the text immediately after eq (4.34). OLS therefore
    ignores both the correlation among the gamma-hat(h_m) and their unequal
    dispersion. Schabenberger & Gotway report Zimmerman and Zimmerman's
    (1991) finding that OLS and WLS perform "more or less equally well", and
    that the greater loss of efficiency comes from ignoring the correlations
    rather than from preferring OLS to WLS -- so this is a serviceable
    estimator, not a strawman.

    Parameters
    ----------
    empirical_variogram : mapping or array-like
        Either a mapping carrying ``lags``, ``gamma`` and optionally
        ``counts``, or an (n_lags, 2) / (n_lags, 3) array of those columns.
    variogram_model : {"exponential", "gaussian", "spherical"}
        Which parametric family of Sec. 4.3 to fit.

    Returns
    -------
    RichResult
        Keys: ``nugget``, ``partial_sill``, ``sill``, ``range``,
        ``objective``, ``converged``, ``n_lags``, ``fitted``.

    References
    ----------
    Schabenberger Ch 4, Sec 4.5.1
    """
    lags, ghat, counts = as_empirical_variogram(empirical_variogram)
    nugget, sill, rng, obj, ok = fit_semivariogram(
        lags, ghat, counts, model=variogram_model, kind="ols")
    fitted = semivariogram(lags, nugget, sill, rng, variogram_model)
    return RichResult(
        title="OLS semivariogram fit",
        summary_lines=[("nugget", nugget), ("partial sill", sill),
                       ("range", rng), ("residual sum of squares", obj)],
        payload={"nugget": nugget, "partial_sill": sill, "sill": nugget + sill,
                 "range": rng, "objective": obj, "converged": ok,
                 "n_lags": int(np.size(lags)), "fitted": fitted,
                 "model": variogram_model, "method": "ordinary least squares"},
    )


def cheatsheet():
    return "spols: OLS fit of a semivariogram model (Schabenberger Sec 4.5.1)"
