# SPDX-License-Identifier: AGPL-3.0-or-later
"""The spatio-temporal semivariogram."""

import numpy as np

from ._richresult import RichResult
from ._schab_st import (conditional_spatial_semivariogram,
                        empirical_st_semivariogram,
                        semivariogram_from_covariance, st_wls_objective)

__all__ = ["schabenberger_st_variogram"]


def schabenberger_st_variogram(coords, times, z, n_space_bins=10,
                               n_time_bins=5, max_dist=None, max_time=None,
                               at_time=None, model_fn=None):
    """Empirical spatio-temporal semivariogram, Sec. 9.4.

    For a stationary spatio-temporal process the semivariogram relates to the
    covariance function exactly as in the purely spatial case:

        gamma(h,k) = 0.5 Var[Z(s,t) - Z(s+h,t+k)]
                   = Var[Z(s,t)] - Cov[Z(s,t), Z(s+h,t+k)]
                   = C(0,0) - C(h,k).

    The estimator is the spatio-temporal Matheron, eq (9.18),

        gamma_hat(h,k) = 1 / (2 |N(h,k)|) sum_{N(h,k)} {Z(s_i,t_i) - Z(s_j,t_j)}^2,

    where N(h,k) is the set of pairs within spatial distance h AND time lag k
    of each other, and |N(h,k)| counts the distinct pairs.

    Two things the text insists on, and which the implementation follows.

    The lag TOLERANCES in space and time must be chosen separately. Data are
    generally irregular in both, and a single tolerance cannot give a
    sufficient number of pairs at each spatio-temporal lag; ``n_space_bins``
    and ``n_time_bins`` are independent for that reason. The pair counts are
    returned alongside the estimates so thin cells are visible rather than
    inferred.

    And (9.18) is a JOINT estimator, not a conditional one. Passing
    ``at_time`` additionally returns the conditional spatial semivariogram of
    eq (9.19), gamma_hat_t(h), which is what a two-stage analysis uses. They
    are different quantities. Sec. 9.1 sets out why the two-stage route is
    weaker: time points with too few data contribute nothing, combining
    per-time estimates needs the temporal correlation between the statistics,
    and predictions from separate analyses cannot interpolate in both
    dimensions.

    Cells containing no pairs are returned as NaN with a count of zero, never
    filled with a value. An unestimated semivariogram and a zero
    semivariogram are different claims.

    Parameters
    ----------
    coords : array-like, shape (n, d)
    times : array-like, shape (n,)
    z : array-like, shape (n,)
    n_space_bins, n_time_bins : int
        Numbers of spatial and temporal lag classes -- the two tolerances.
    max_dist, max_time : float, optional
        Largest lags retained. Default to half the observed maxima.
    at_time : float, optional
        If given, also compute the conditional (9.19) estimator at that time.
    model_fn : callable, optional
        gamma(h, k; theta). If given, the weighted least squares criterion of
        Sec. 9.4 is evaluated against the empirical surface.

    Returns
    -------
    RichResult
        Keys: ``st_variogram``, ``counts``, ``space_lags``, ``time_lags``,
        ``n_pairs``, and optionally ``conditional`` and ``wls_objective``.

    References
    ----------
    Schabenberger & Gotway (2005), Sec. 9.4, eqs (9.18)-(9.19).
    """
    emp = empirical_st_semivariogram(coords, times, z,
                                     n_space_bins=n_space_bins,
                                     n_time_bins=n_time_bins,
                                     max_dist=max_dist, max_time=max_time)
    counts = emp["counts"]
    filled = int(np.count_nonzero(counts))
    payload = {
        "st_variogram": emp["gamma"],
        "counts": counts,
        "space_lags": emp["space_lags"],
        "time_lags": emp["time_lags"],
        "space_edges": emp["space_edges"],
        "time_edges": emp["time_edges"],
        "n_pairs": int(counts.sum()),
        "n_cells": int(counts.size),
        "n_cells_estimated": filled,
    }
    lines = [("pairs used", payload["n_pairs"]),
             ("cells estimated", f"{filled} of {counts.size}"),
             ("smallest cell count", int(counts[counts > 0].min())
              if filled else 0)]
    if filled < counts.size:
        payload["warning"] = (
            f"{counts.size - filled} of {counts.size} lag cells contain no "
            f"pairs and are NaN; widen the tolerances or reduce the bin counts")

    if at_time is not None:
        payload["conditional"] = conditional_spatial_semivariogram(
            coords, times, z, at_time=at_time, n_bins=n_space_bins,
            max_dist=max_dist)
        payload["conditional_note"] = (
            "eq (9.19) is the CONDITIONAL spatial semivariogram at one time, "
            "used by two-stage analyses; it is not comparable with the joint "
            "estimator (9.18) above")
        lines.append(("conditional at t", at_time))

    if model_fn is not None:
        payload["wls_objective"] = st_wls_objective(emp, model_fn)
        hh, kk = np.meshgrid(emp["space_lags"], emp["time_lags"],
                             indexing="ij")
        payload["fitted"] = np.asarray(model_fn(hh, kk), dtype=float)
        lines.append(("WLS objective", payload["wls_objective"]))

    return RichResult(title="Spatio-temporal semivariogram",
                      summary_lines=lines, payload=payload)


def st_variogram_from_model(spatial_h, temporal_u, cov_fn):
    """gamma(h,k) = C(0,0) - C(h,k), the Sec. 9.4 identity."""
    return semivariogram_from_covariance(spatial_h, temporal_u, cov_fn)


def cheatsheet():
    return ("spstvg: spatio-temporal semivariogram (Sec. 9.4) -- joint "
            "estimator (9.18), conditional (9.19), and the WLS criterion")
