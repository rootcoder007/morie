# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inverse distance weighted interpolation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["schabenberger_idw"]


def schabenberger_idw(coords, z, target, power=2.0):
    """Inverse distance weighted interpolation.

    Bivand, Pebesma and Gomez-Rubio (2013), Sec. 8.3.1 -- NOT a Schabenberger
    method. Inverse distance weighting appears in Statistical Methods for
    Spatial Data Analysis only in the subject index, so this module is
    grounded in its own primary source.

        Z_hat(s0) = sum_i w(s_i) Z(s_i) / sum_i w(s_i),
        w(s_i)    = ||s_i - s0||^-p,

    with p "an inverse distance weighting power, defaulting to 2".

    Three properties the text states, all asserted in the suites:

    * "If s0 coincides with an observation location, the observed value is
      returned to avoid infinite weights." So IDW is an exact interpolator,
      by an explicit rule rather than by a limit.
    * "for large values IDW converges to the one-nearest-neighbour
      interpolation".
    * "inverse distance does not provide prediction error variances" -- the
      reference implementation returns NA for the variance. This module
      returns None rather than inventing a number, because a fabricated
      variance is worse than an absent one.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Observation locations.
    z : array-like, shape (n,)
        Observed values.
    target : array-like, shape (d,) or (m, d)
        Prediction location(s).
    power : float
        The inverse distance power p. Must be non-negative; p = 0 gives the
        unweighted mean, which is the correct limit and not a special case.

    Returns
    -------
    RichResult
        Keys: ``prediction``, ``variance`` (always None), ``weights``,
        ``power``, ``exact_hits``.

    References
    ----------
    Bivand, R. S., Pebesma, E., and Gomez-Rubio, V. (2013) Applied Spatial
    Data Analysis with R, 2nd ed., Springer. Sec. 8.3.1 "Inverse Distance
    Weighted Interpolation".
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    target = np.atleast_2d(np.asarray(target, dtype=float))
    power = float(power)
    if power < 0.0:
        raise ValueError("`power` must be non-negative")
    if coords.shape[0] != z.size:
        raise ValueError("`coords` and `z` must have the same number of rows")
    if coords.shape[1] != target.shape[1]:
        raise ValueError("`coords` and `target` must have the same dimension")

    dist = np.linalg.norm(coords[None, :, :] - target[:, None, :], axis=-1)
    preds = np.empty(target.shape[0])
    weights = np.zeros_like(dist)
    exact = np.zeros(target.shape[0], dtype=bool)
    for j in range(target.shape[0]):
        d = dist[j]
        hit = d == 0.0
        if np.any(hit):
            # The coincidence rule, stated in the text: return the observed
            # value rather than forming an infinite weight.
            exact[j] = True
            preds[j] = float(z[hit].mean())
            weights[j, hit] = 1.0 / int(hit.sum())
            continue
        w = d ** (-power)
        weights[j] = w / w.sum()
        preds[j] = float(weights[j] @ z)
    single = target.shape[0] == 1
    return RichResult(
        title="Inverse distance weighted interpolation",
        summary_lines=[("power", power), ("n targets", target.shape[0]),
                       ("exact hits", int(exact.sum()))],
        payload={"prediction": float(preds[0]) if single else preds,
                 "variance": None,
                 "weights": weights[0] if single else weights,
                 "power": power, "exact_hits": bool(exact[0]) if single else exact,
                 "method": "inverse distance weighting"},
    )


def cheatsheet():
    return "spmidw: inverse distance weighting (Bivand et al. 2013, Sec 8.3.1)"
