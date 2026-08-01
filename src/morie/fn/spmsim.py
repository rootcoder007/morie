# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multiscale GWR (MGWR): one bandwidth per covariate."""

import numpy as np

from ._richresult import RichResult
from ._schab_gwr import mgwr_backfit, pairwise_distances

__all__ = ["schabenberger_mgwr_bandwidth"]


def schabenberger_mgwr_bandwidth(x, y, coords, kernel="gaussian",
                                 criterion="aicc", adaptive=False, tol=1e-5,
                                 max_iter=200, rss_score=False,
                                 bws_same_times=5, init_bandwidth=None,
                                 standardize=True):
    """Multiscale GWR: fit each covariate at its own spatial scale.

    Ordinary GWR gives every covariate the same bandwidth, which asserts
    that every relationship in the model varies over space at the same rate.
    MGWR drops that assumption: each ``beta_k(s)`` gets its own bandwidth
    ``h_k``, so a covariate whose effect is near-constant can take a wide
    kernel while one that genuinely varies locally takes a narrow one.

    Estimation is GAM backfitting. Start from an ordinary GWR fit with a
    single bandwidth. Then sweep the covariates: for each ``j``, form the
    partial residual ``XB[:, j] + err``, select a bandwidth for a
    *univariate* GWR of that partial residual on ``x_j`` alone, refit, and
    carry the updated residual into the next covariate of the same sweep.
    Sweep until the score of change (SOC) falls below ``tol``:

    ``rss_score=False`` (SOC-f, the default)
        ``sqrt( (sum (XB_new - XB)^2 / n) / sum_i (sum_j XB_new[i,j])^2 )``
    ``rss_score=True`` (SOC-RSS)
        ``|rss_new - rss| / rss_new``

    If the bandwidth vector repeats unchanged for ``bws_same_times``
    consecutive sweeps the search is frozen and only the coefficients are
    refitted, which is what keeps the cost tolerable -- each sweep otherwise
    runs a full bandwidth search per covariate.

    ``standardize`` defaults to True on the authority of Fotheringham,
    Oshan & Li (2024) Sec. 2.3.3.2, which states that comparing the
    covariate-specific bandwidths to one another *requires* y and each
    column of X to be standardized first, and Sec. 6.3, which records that
    in the authors' own software standardization is a default that "has to
    be actively turned off". Constant columns are left alone. Coefficients
    come back on the standardized scale, with the centres and scales in the
    payload; ``fitted`` and ``resid`` are converted back to the units of
    ``y``. Measured on the two-scale fixture in the test suite,
    standardizing did NOT change which covariate got the wider bandwidth
    (6 of 8 seeds either way) -- that fixture's covariates are already of
    comparable magnitude, so the correction has nothing to bite on. It
    matters when covariates are measured on genuinely different scales.

    MGWR inference is NOT provided: the covariate-specific hat matrices
    ``R_k`` (2024 eqs (2.40)-(2.42)), ``ENP_k = tr(R_k)`` (eq (2.43)) and
    the adjusted alpha of eq (2.45), all after Yu et al. (2020), are not
    implemented. This returns bandwidths and coefficients.

    MGWR postdates Schabenberger & Gotway (2005) and is not in that book;
    it is included here because the shelf names it.

    Parameters
    ----------
    x : array-like, shape (n, p)
        Design matrix. The intercept, if present, is treated as a covariate
        like any other and gets its own bandwidth.
    y : array-like, shape (n,)
    coords : array-like, shape (n, 2)
    kernel : {'gaussian', 'bisquare', 'tricube', 'boxcar'}
    criterion : {'aicc', 'cv', 'aic'}
        Passed to each inner univariate bandwidth search.
    adaptive : bool, default False
    tol : float, default 1e-5
    max_iter : int, default 200
    rss_score : bool, default False
    bws_same_times : int, default 5
    init_bandwidth : float, optional
        Skip the initial single-bandwidth search and start here.
    standardize : bool, default True
        Standardize y and every non-constant column of x to mean 0,
        variance 1 before calibrating.

    Returns
    -------
    RichResult
        Keys: ``bandwidths`` (one per covariate), ``local_coefficients``,
        ``fitted``, ``resid``, ``rss``, ``bandwidth_gwr`` (the
        single-bandwidth starting point), ``bandwidth_history``,
        ``score_history``, ``n_iter``, ``converged``.

    References
    ----------
    Fotheringham, A. S., Yang, W. & Kang, W. (2017). Multiscale
    geographically weighted regression (MGWR). Annals of the American
    Association of Geographers, 107(6):1247-1265.
    doi:10.1080/24694452.2017.1352480. Read: eq. (9) SOC-RSS, eq. (10)
    SOC-f, the back-fitting algorithm of Figure 1, GWR estimates as the
    initialisation, and SOC-f <= 1e-5 as the termination criterion.
    Fotheringham, A. S., Oshan, T. M. & Li, Z. (2024). Multiscale
    Geographically Weighted Regression: Theory and Practice, 1st ed. CRC
    Press. doi:10.1201/9781003435464. Sec. 2.3.2 eqs (2.38)-(2.39) restate
    the SOC; Sec. 2.3.3.2 and Sec. 6.3 require standardization.
    Oshan, T. M., Li, Z., Kang, W., Wolf, L. J. & Fotheringham, A. S.
    (2019), ``mgwr``, ``mgwr/search.py``, function ``multi_bw`` -- the
    authors' own implementation, used to settle points of ordering the
    printed algorithm leaves open.
    Schabenberger, O. & Gotway, C. A. (2005), Sec. 6.1.3.1, pp. 316-317, for
    the single-scale GWR this generalises.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    res = mgwr_backfit(y, x, coords, kernel=kernel, criterion=criterion,
                       adaptive=adaptive, tol=tol, max_iter=max_iter,
                       rss_score=rss_score, bws_same_times=bws_same_times,
                       init_bandwidth=init_bandwidth,
                       standardize=standardize)
    bws = res["bandwidths"]
    payload = {
        "bandwidths": bws,
        "local_coefficients": res["params"],
        "fitted": res["fitted"],
        "resid": res["resid"],
        "rss": float(np.sum(res["resid"] ** 2)),
        "bandwidth_gwr": res["bandwidth_gwr"],
        "bandwidth_history": res["bandwidth_history"],
        "score_history": res["score_history"],
        "n_iter": res["n_iter"],
        "converged": res["converged"],
        "at_search_boundary": res["at_search_boundary"],
        "standardized": res["standardized"],
        "y_centre": res["y_centre"],
        "y_scale": res["y_scale"],
        "x_centre": res["x_centre"],
        "x_scale": res["x_scale"],
        "criterion": criterion,
        "kernel": kernel,
        "score_type": "SOC-RSS" if rss_score else "SOC-f",
        "n": int(x.shape[0]),
        "p": int(x.shape[1]),
    }
    if res["at_search_boundary"]:
        payload["warning"] = (
            "every bandwidth sits at the top of the search interval and the "
            "backfit stopped after "
            f"{res['n_iter']} sweep(s): the SOC measures how much the fit "
            "MOVED, so a first sweep that changes nothing scores as "
            "converged. No scale separation was found -- rerun from a "
            "narrower init_bandwidth before reading anything into these "
            "bandwidths")
    if not res["converged"]:
        payload["warning"] = (
            f"backfitting hit max_iter={max_iter} with SOC="
            f"{res['score_history'][-1]:.3e} still above tol={tol}; the "
            "bandwidths are the last sweep's, not a converged optimum")
    return RichResult(
        title="Multiscale GWR bandwidths",
        summary_lines=[
            ("bandwidths", np.round(bws, 4).tolist()),
            ("single-bandwidth GWR", res["bandwidth_gwr"]),
            ("sweeps", res["n_iter"]),
            ("converged", res["converged"]),
            ("at search boundary", res["at_search_boundary"]),
            (payload["score_type"], res["score_history"][-1]),
        ],
        payload=payload,
    )


def cheatsheet():
    return ("spmsim: multiscale GWR -- one bandwidth per covariate by GAM "
            "backfitting (Fotheringham, Yang & Kang 2017)")


# CANONICAL TEST
# Two covariates acting at genuinely different scales -- an intercept whose
# level drifts slowly across the domain and a slope that oscillates -- must
# come back with two clearly different bandwidths, and MGWR's residual sum
# of squares must not exceed that of the single-bandwidth GWR it started
# from.  See scripts/audit/schab_gwr_verify.py.
