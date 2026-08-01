# SPDX-License-Identifier: AGPL-3.0-or-later
"""GWR bandwidth selection by cross-validation or corrected AIC."""

import numpy as np

from ._richresult import RichResult
from ._schab_gwr import (aic_from_parts, aicc_from_parts, cv_score, gwr_fit,
                         pairwise_distances, select_bandwidth)

__all__ = ["schabenberger_gwr_bandwidth"]


def schabenberger_gwr_bandwidth(x, y, coords, kernel="gaussian",
                                criterion="cv", adaptive=False, bounds=None,
                                tol=1e-4):
    """Choose the GWR bandwidth.

    The bandwidth, not the kernel, is what decides a GWR fit -- "the choice
    of a bandwidth is more important than the shape of the kernel". As it
    grows the local models converge on the single global OLS fit; as it
    shrinks each local fit sees fewer neighbours and spends more degrees of
    freedom. Both criteria implemented here trade those two off.

    ``criterion='cv'``
        Leave-one-out cross-validation, ``sum_i (y_i - yhat_{-i})^2``. The
        local model at ``i`` is fitted with its own weight ``w_ii`` forced
        to zero, so ``y_i`` never contributes to predicting itself. This is
        what Bivand et al. (2013, Sec. 9.4.3) describe as the usual choice.

    ``criterion='aicc'``
        The corrected AIC of Hurvich, Simonoff & Tsai (1998) as adopted for
        GWR,

            AICc = 2n log(sigma_hat) + n log(2 pi)
                   + n (n + tr(S)) / (n - 2 - tr(S)),

        with ``sigma_hat^2 = y'(I-S)'(I-S)y / n`` and ``S`` the hat matrix
        of Sec. 6.1.3.1, p. 317. ``criterion='aic'`` gives the uncorrected
        form ``... + n + tr(S)``.

    Three different residual variances appear here, because the sources use
    three and they are not interchangeable:

    ``sigma2``
        ``RSS / n``, the maximum-likelihood estimate.  Fotheringham et al.
        (2002) eq. (4.23) says outright that this, and not the other, is
        what the AIC and AICc take.
    ``sigma2_gwr``
        ``RSS / (n - 2 v1 + v2)`` with ``v1 = tr(S)`` and ``v2 = tr(S'S)``
        -- eq. (2.16), whose denominator is the effective residual degrees
        of freedom.  This is the one the local standard errors of eq. (2.15)
        are built from, reported here as ``se_params``.
    ``sigma2_cressie``
        ``RSS / tr{(I-L)(I-L)'}`` -- Schabenberger & Gotway p. 317, after
        Cressie (1998).

    Substituting one for another silently shifts either the criterion or the
    standard errors, so all three are reported rather than one being chosen
    on the caller's behalf.

    Search is by golden section -- which is the method the source itself
    names (Fotheringham et al. 2002 p. 60, citing Greig 1980) -- over
    ``[diagonal/1000, diagonal]`` of the coordinate bounding box, the
    interval ``spgwr::gwr.sel`` uses. With
    ``adaptive=True`` the bandwidth is instead a neighbour count and the
    criterion is evaluated on every integer in ``[2, n]``.

    Parameters
    ----------
    x : array-like, shape (n, p)
        Design matrix. Include an intercept column if one is wanted.
    y : array-like, shape (n,)
    coords : array-like, shape (n, 2)
    kernel : {'gaussian', 'bisquare', 'tricube', 'boxcar'}
    criterion : {'cv', 'aicc', 'aic'}
    adaptive : bool, default False
    bounds : tuple, optional
        Override the search interval.
    tol : float, default 1e-4
        Golden-section stopping width. Ignored when ``adaptive=True``.

    Returns
    -------
    RichResult
        Keys: ``optimal_bandwidth``, ``score``, ``criterion``, ``bounds``,
        ``adaptive``, ``kernel``, plus the fit at the optimum -- ``tr_S``,
        ``effective_parameters``, ``sigma2``, ``sigma2_cressie``, ``rss``,
        ``cv``, ``aicc``, ``aic``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 6.1.3.1, pp. 316-317 --
    the model, the hat matrix and Cressie's residual variance. The book
    gives no bandwidth-selection criterion and defers to Fotheringham et al.
    Bivand, R. S., Pebesma, E. & Gomez-Rubio, V. (2013). Applied Spatial
    Data Analysis with R, 2nd ed. Springer, Sec. 9.4.3, p. 318: a Gaussian
    kernel "with a fixed bandwidth chosen by leave-one-out cross-validation".
    Hurvich, C. M., Simonoff, J. S. & Tsai, C.-L. (1998). Smoothing
    parameter selection in nonparametric regression using an improved
    Akaike information criterion. JRSS B, 60:271-293. The AICc.
    Fotheringham, A. S., Brunsdon, C. & Charlton, M. E. (2002).
    Geographically Weighted Regression: The Analysis of Spatially Varying
    Relationships. Wiley, Chichester. Read directly: eq. (2.31) the CV
    score with "the observations for point i omitted from the calibration
    process"; eq. (2.33) = eq. (4.21) the AICc; eq. (4.22) the AIC;
    eq. (4.23) fixing sigma-hat squared as RSS/n for both; eqs. (2.17) and
    (2.18) v1 = tr(S) and v2 = tr(S'S), with "2v1 - v2 ... the effective
    number of parameters"; eq. (2.16) the inference variance; eqs. (2.14)
    and (2.15) the parameter variance; eq. (2.20) the hat-matrix row.
    Golden section is the book's own named search method (p. 60, after
    Greig 1980).  Independently checked against ``spgwr``'s published NY8
    output.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    sel = select_bandwidth(y, x, coords, kernel=kernel, criterion=criterion,
                           adaptive=adaptive, bounds=bounds, tol=tol)
    bw = sel["bandwidth"]
    D = pairwise_distances(coords)
    fit = gwr_fit(y, x, D, bw, kernel, adaptive)
    n, sigma2, tr_S = fit["n"], fit["sigma2"], fit["tr_S"]

    payload = {
        "optimal_bandwidth": bw,
        "score": sel["score"],
        "criterion": criterion,
        "bounds": sel["bounds"],
        "adaptive": bool(adaptive),
        "kernel": kernel,
        "tr_S": tr_S,
        "tr_STS": fit["tr_STS"],
        "effective_parameters": fit["effective_parameters"],
        "rss": fit["rss"],
        "sigma2": sigma2,
        "sigma2_cressie": fit["sigma2_cressie"],
        # eq (2.16), denominator n - 2v1 + v2, with the local standard
        # errors of eq (2.15) that go with it
        "sigma2_gwr": fit["sigma2_gwr"],
        "edf_resid": fit["edf_resid"],
        "se_params": fit["se_params"],
        "v1": fit["v1"],
        "v2": fit["v2"],
        "cv": cv_score(y, x, D, bw, kernel, adaptive),
        "aicc": aicc_from_parts(n, sigma2, tr_S),
        "aic": aic_from_parts(n, sigma2, tr_S),
        "n": n,
    }
    return RichResult(
        title=f"GWR bandwidth selection ({criterion})",
        summary_lines=[
            ("criterion", criterion),
            ("bandwidth", bw),
            ("kernel", kernel + (" (adaptive)" if adaptive else " (fixed)")),
            ("tr(S)", tr_S),
            ("effective parameters", fit["effective_parameters"]),
            ("AICc", payload["aicc"]),
            ("CV", payload["cv"]),
        ],
        payload=payload,
    )


def cheatsheet():
    return ("spgwrb: GWR bandwidth by leave-one-out CV or AICc; fixed or "
            "adaptive, golden-section search over the bounding-box diagonal")
