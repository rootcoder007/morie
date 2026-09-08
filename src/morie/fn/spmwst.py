# SPDX-License-Identifier: AGPL-3.0-or-later
"""Moving-window kriging with locally re-estimated semivariograms."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_nonstat import haas_window, moving_window_krige

__all__ = ["schabenberger_moving_window"]


def schabenberger_moving_window(coords, z, window_size=None, targets=None,
                                min_sites=35, step=5, n_lags=10,
                                local_variogram=True, local_mean=False):
    """Haas's moving-window approach, Sec. 8.3.1.

    Local kriging restricts the solve to a neighbourhood but keeps ONE global
    covariance model -- "all n data points contribute to the estimation of
    theta in local kriging" (p. 425). The moving-window method of Haas (1990,
    1995) goes further and re-estimates the semivariogram *inside each
    window*, so every prediction location carries its own ``theta_i``. That
    distinction is the content of the section, and ``local_variogram``
    selects between them.

    Window size follows the rule on p. 426: enlarge a circle about the
    prediction site until at least ``min_sites`` sites are inside, then add
    ``step`` at a time until every lag class holds at least one pair and the
    local semivariogram fit converges.

    The book's cautions are returned rather than hidden: a predictor that
    excludes observed sites "is no longer best", and windows that change with
    prediction location can introduce "spurious discontinuities".

    Parameters
    ----------
    coords : array-like, shape (n, d)
    z : array-like, shape (n,)
    window_size : float, optional
        A fixed radius. When given it overrides Haas's adaptive rule, and the
        number of sites actually captured per window is reported so a window
        too small to fit a semivariogram is visible.
    targets : array-like, optional
        Prediction locations. Defaults to the observed coordinates.
    min_sites, step : int
    n_lags : int
    local_variogram : bool, default True
    local_mean : bool, default False

    Returns
    -------
    RichResult
        Keys: ``local_variograms`` (per-window sill and range), ``prediction``,
        ``window_sizes``, ``converged``, ``global_sill``, ``global_range``,
        ``theta_is_global``, ``caveats``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005), Sec. 8.3.1, pp. 425-426.
    Haas, T. C. (1990), Lognormal and moving window methods of estimating
    acid deposition, JASA 85:950-963; Haas, T. C. (1995), Local prediction of
    a spatio-temporal process with an application to wet sulfate deposition,
    JASA 90:1189-1199.
    """
    s = np.asarray(coords, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    z = np.asarray(z, dtype=float).ravel()
    if s.shape[0] != z.size:
        raise ValueError(f"{s.shape[0]} coordinates but {z.size} observations")
    tg = s if targets is None else np.asarray(targets, dtype=float)
    if tg.ndim == 1:
        tg = tg.reshape(-1, s.shape[1])

    fixed_counts = None
    if window_size is not None:
        w = float(window_size)
        if w <= 0:
            raise ValueError("window_size must be positive")
        d = np.linalg.norm(s[None, :, :] - tg[:, None, :], axis=-1)
        fixed_counts = (d <= w).sum(axis=1)
        min_sites = max(2, int(fixed_counts.min()))

    res = moving_window_krige(s, z, tg, min_sites=min_sites, step=step,
                              n_lags=n_lags, local_mean=local_mean,
                              local_variogram=local_variogram)
    payload = dict(res)
    payload["local_variograms"] = np.column_stack(
        [res["local_sill"], res["local_range"]])
    payload["targets"] = tg
    if fixed_counts is not None:
        payload["fixed_window_size"] = float(window_size)
        payload["fixed_window_counts"] = fixed_counts
        if int(fixed_counts.min()) < min_sites:
            payload["warning"] = (
                f"the requested window holds as few as {int(fixed_counts.min())} "
                "sites, below the 35 that Sec. 8.3.1 sets as the starting "
                "point for a reliable local semivariogram")
    lines = [("targets", tg.shape[0]),
             ("theta re-estimated per window", not res["theta_is_global"]),
             ("median window size", float(np.median(res["window_sizes"]))),
             ("windows that converged", int(np.sum(res["converged"])))]
    return RichResult(title="Moving-window local semivariograms",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spmwst: Haas moving-window kriging with per-window semivariogram "
            "re-estimation (Sec. 8.3.1)")
