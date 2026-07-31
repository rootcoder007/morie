# SPDX-License-Identifier: AGPL-3.0-or-later
"""Weighted least squares fitting of a semivariogram model."""

import numpy as np

from ._richresult import RichResult
from ._schab_fit import as_empirical_variogram, fit_semivariogram
from ._schab_vario import semivariogram

__all__ = ["schabenberger_wls_variogram"]


def schabenberger_wls_variogram(empirical_variogram, variogram_model="exponential"):
    """Fit a semivariogram model by Cressie's weighted least squares.

    Minimises eq (4.34),

        sum_m |N(h_m)| / (2 gamma(h_m; theta)^2)
              * {gamma_hat(h_m) - gamma(h_m; theta)}^2,

    which is the generalized sum of squares (4.31) with R(theta) replaced by
    the diagonal matrix W(theta) whose entries are Cressie's (1985)
    approximation (4.33), Var[gamma_hat(h_m)] ~ 2 gamma(h_m,theta)^2/|N(h_m)|.
    Dividing through by 2 gamma^2 gives the equivalent and more familiar
    (1/2) sum_m |N(h_m)| [gamma_hat(h_m)/gamma(h_m;theta) - 1]^2; the two
    differ by a constant factor and so share a minimiser.

    The weights are functions of theta, so this is a re-weighted rather than
    a fixed-weight fit -- the text is explicit that updates to theta must be
    followed by updates to W(theta). Note the book's own caveat: because the
    off-diagonal entries of R(theta) are appreciable, WLS is a poor
    approximation to (4.31), and the efficiency lost by ignoring those
    correlations exceeds anything gained over OLS.

    Parameters
    ----------
    empirical_variogram : mapping or array-like
        Either a mapping carrying ``lags``, ``gamma`` and optionally
        ``counts``, or an (n_lags, 2) / (n_lags, 3) array of those columns.
        Counts matter here: they are the |N(h_m)| of eq (4.34).
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
        lags, ghat, counts, model=variogram_model, kind="wls")
    fitted = semivariogram(lags, nugget, sill, rng, variogram_model)
    return RichResult(
        title="WLS semivariogram fit",
        summary_lines=[("nugget", nugget), ("partial sill", sill),
                       ("range", rng), ("weighted sum of squares", obj)],
        payload={"nugget": nugget, "partial_sill": sill, "sill": nugget + sill,
                 "range": rng, "objective": obj, "converged": ok,
                 "n_lags": int(np.size(lags)), "fitted": fitted,
                 "model": variogram_model, "method": "weighted least squares"},
    )


def cheatsheet():
    return "spwls: Cressie WLS fit of a semivariogram model (Schabenberger Sec 4.5.1)"
