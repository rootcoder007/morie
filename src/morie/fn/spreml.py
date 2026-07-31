# SPDX-License-Identifier: AGPL-3.0-or-later
"""Restricted maximum likelihood (REML) for semivariogram parameters."""

import numpy as np

from ._richresult import RichResult
from ._schab_reml import fit_reml

__all__ = ["schabenberger_reml_variogram"]


def schabenberger_reml_variogram(coords, z, X=None, variogram_model="exponential"):
    """Estimate covariance parameters by restricted maximum likelihood.

    Follows Sec. 4.5.2 and Sec. 5.5.3 rather than improvising:

    * The matrix of error contrasts K is eliminated. Sec. 5.5.3 quotes
      Searle et al. (1992, pp. 451-452) for
      ``K'(K Sigma K')^-1 K = Sigma^-1 - Sigma^-1 X Omega X' Sigma^-1``,
      which reduces ``Z'K'(K Sigma K')^-1 KZ`` to ``r' Sigma^-1 r`` and gives
      the objective ``ln|Sigma| + ln|X' Sigma^-1 X| + r' Sigma^-1 r
      + (n-k) ln(2 pi)``. No contrast matrix is ever formed. Harville (1977)
      notes that admissible choices of K change the objective only by a
      constant free of theta and beta, so the minimiser is unaffected -- the
      test suite checks that constant is in fact constant.
    * A scale parameter is profiled out, eq (5.49). With
      ``Sigma(theta) = sigma^2 Sigma(theta*)`` the REML estimator is
      ``sigma^2 = r' Sigma(theta*)^-1 r / (n - k)``, leaving just the nugget
      RATIO and the range to be optimised -- two parameters, not three.
    * Sec. 5.5.2 names the optimiser: "Newton-Raphson, Quasi-Newton, or some
      other suitable algorithm". The quasi-Newton branch is used, driven by
      the exact analytic gradient, so no finite-difference step enters and
      both language arms execute the same steps.

    There is no REML estimator of the mean. The text is explicit that
    beta_reml "is simply an EGLS estimator evaluated at theta_reml", which is
    what ``mean`` reports.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Sampling locations.
    z : array-like, shape (n,)
        Observed values.
    X : array-like, shape (n, p), optional
        Design matrix for the mean. Defaults to an intercept, the
        E[Z(s)] = mu case worked in Sec. 4.5.2.
    variogram_model : {"exponential", "gaussian", "spherical"}
        Which parametric family of Sec. 4.3 to fit.

    Returns
    -------
    RichResult
        Keys: ``nugget``, ``partial_sill``, ``sill``, ``range``,
        ``nugget_ratio``, ``mean``, ``neg2_restricted_loglik``,
        ``converged``, ``n``, ``n_contrasts``.

    Notes
    -----
    The nugget is weakly identified at modest sample sizes -- it trades off
    against microscale variation -- so it is the least reliable of the three
    even when the range and the mean are recovered well.

    References
    ----------
    Schabenberger Ch 4, Sec 4.5.2; Ch 5, Sec 5.5.3
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    n = z.size
    if coords.shape[0] != n:
        raise ValueError("`coords` and `z` must have the same number of rows")
    X = np.ones((n, 1)) if X is None else np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != n:
        raise ValueError("`X` must have one row per observation")

    fit = fit_reml(coords, z, X, model=variogram_model)
    beta = fit["beta"]
    return RichResult(
        title="REML covariance-parameter estimates",
        summary_lines=[("nugget", fit["nugget"]),
                       ("partial sill", fit["partial_sill"]),
                       ("range", fit["range"]),
                       ("-2 restricted logL", fit["neg2_restricted_loglik"])],
        payload={"nugget": fit["nugget"], "partial_sill": fit["partial_sill"],
                 "sill": fit["nugget"] + fit["partial_sill"],
                 "range": fit["range"], "nugget_ratio": fit["nugget_ratio"],
                 "mean": beta if beta.size > 1 else float(beta[0]),
                 "neg2_restricted_loglik": fit["neg2_restricted_loglik"],
                 "converged": fit["converged"], "n": int(n),
                 "n_contrasts": int(n - X.shape[1]),
                 "model": variogram_model,
                 "method": "restricted maximum likelihood"},
    )


def cheatsheet():
    return "spreml: REML covariance-parameter estimates (Schabenberger Sec 4.5.2)"
