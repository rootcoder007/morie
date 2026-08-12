r"""Bandwidth selection for geographically weighted regression.

Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002) *Geographically
Weighted Regression: The Analysis of Spatially Varying Relationships*,
Wiley.

GWR fits a separate regression at every location, weighting observations
by their distance from it:

.. math::

   \hat\beta(s_i) = \bigl(X' W(s_i) X\bigr)^{-1} X' W(s_i) y ,

so the whole method turns on the bandwidth of :math:`W`. It is the only
parameter that matters and it cannot be guessed: too wide and every local
fit is the global one, too narrow and each fit chases its own
neighbourhood. This module is the calibration step -- choosing it from the
data by one of the book's three criteria.

.. math::

   CV(h) &= \sum_i \bigl(y_i - \hat y_{\ne i}(h)\bigr)^2 \\
   AIC(h) &= 2n \log\hat\sigma + n\log 2\pi + n + \mathrm{tr}(S)
             \tag{4.22} \\
   AIC_c(h) &= 2n \log\hat\sigma + n\log 2\pi
               + n\,\frac{n + \mathrm{tr}(S)}{n - 2 - \mathrm{tr}(S)}
             \tag{2.33}

The cross-validation score drops observation :math:`i` from its own local
fit -- without that it would simply choose the smallest bandwidth on
offer, since a narrow kernel reproduces :math:`y_i` from :math:`y_i`. The
AIC forms instead pay for flexibility through the effective number of
parameters :math:`\mathrm{tr}(S)`, where :math:`S` is the hat matrix whose
:math:`i`-th row is :math:`x(s_i)'\{X'W(s_i)X\}^{-1}X'W(s_i)`. AICc is the
default: it carries the small-sample correction, and GWR's whole point is
that local samples are small.

That default is not a preference, it is a requirement. Plain AIC's
penalty is linear in :math:`\mathrm{tr}(S)` while the log-likelihood term
runs away as the fit approaches interpolation, so on the panels anchored
here ``criterion="aic"`` drives the bandwidth to the bottom of the search
interval -- 0.018 against AICc's 1.59, at a "better" score of -6805. It
is available because the book prints it, and it is not the default for
this reason. Check ``at_boundary`` before believing any bandwidth.

A bandwidth may be **fixed** (a distance, in coordinate units) or
**adaptive** (a neighbour count, so the kernel is wide where data are
sparse and narrow where they are dense). Both are here, and adaptive is
searched over integers because that is what it is.

Two search routes:

``"golden"`` (default for a fixed bandwidth)
    Golden-section minimisation on the search interval, which for a fixed
    bandwidth defaults to the bounding-box diagonal down to a thousandth
    of it.
``"grid"``
    Evaluate the whole grid. Slower, and worth it when the profile is not
    unimodal -- which happens, and which golden section will not tell you
    about. The profile is returned either way so it can be looked at.

The result also reports the global OLS fit's AICc alongside the GWR one,
because the first question about a GWR is whether it earns its extra
parameters at all: the book's rule of thumb is that a difference of three
or more is worth taking seriously.
"""

import math

from . import _array_core as np
from . import _schab_gwr as _gwr

from ._richresult import RichResult

__all__ = [
    "gwr_bandwidth_select",
    "gwr_calibrate",
    "bandwidth_profile",
    "global_ols_aicc",
]

_KERNELS = ("gaussian", "bisquare", "tricube", "boxcar")
_CRITERIA = ("aicc", "cv", "aic")


def _prepare(y, X, coords):
    y = [float(v) for v in np.atleast_1d(np.asarray(y, dtype=float))]
    n = len(y)
    if n < 3:
        raise ValueError("gwrcal: need at least three observations")
    Xr = [list(map(float, r)) for r in np.asarray(X, dtype=float)]
    if len(Xr) != n:
        raise ValueError("gwrcal: X has %d rows for %d responses"
                         % (len(Xr), n))
    p = len(Xr[0])
    if any(len(r) != p for r in Xr):
        raise ValueError("gwrcal: X is ragged")
    C = [list(map(float, r)) for r in np.asarray(coords, dtype=float)]
    if len(C) != n:
        raise ValueError("gwrcal: coords has %d rows for %d observations"
                         % (len(C), n))
    for block, name in ((Xr, "X"), (C, "coords")):
        for r in block:
            for v in r:
                if v != v or v in (float("inf"), float("-inf")):
                    raise ValueError("gwrcal: %s contains a non-finite "
                                     "value" % name)
    for v in y:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("gwrcal: y contains a non-finite value")
    if p >= n:
        raise ValueError("gwrcal: %d columns in X for %d observations "
                         "leaves no residual degrees of freedom"
                         % (p, n))
    return y, Xr, C, n, p


def global_ols_aicc(y, X):
    r"""AICc of the ordinary least squares fit, on the same scale as the
    GWR one (:math:`\mathrm{tr}(S) = p`)."""
    y, Xr, _, n, p = _prepare(y, X, [[0.0]] * len(y))
    Xa = np.asarray(Xr, dtype=float)
    ya = np.asarray(y, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(Xa, ya)
    resid = [y[i] - float(sum(Xr[i][j] * float(beta[j])
                              for j in range(p))) for i in range(n)]
    sigma2 = sum(r * r for r in resid) / n
    if sigma2 <= 0:
        return float("-inf")
    return _gwr.aicc_from_parts(n, sigma2, float(p))


def bandwidth_profile(y, X, coords, kernel="gaussian", criterion="aicc",
                      adaptive=False, bounds=None, n_points=30):
    """The selection criterion over a grid of bandwidths.

    Returned by :func:`gwr_calibrate` so the shape of the profile can be
    inspected rather than assumed unimodal.
    """
    y, Xr, C, n, p = _prepare(y, X, coords)
    D = _gwr.pairwise_distances(C)
    if adaptive:
        lo, hi = (p + 1, n) if bounds is None else bounds
        grid = list(range(int(math.ceil(lo)), int(math.floor(hi)) + 1))
    else:
        lo, hi = _gwr._default_bounds(C) if bounds is None else bounds
        if not hi > lo:
            raise ValueError("gwrcal: the upper bound must exceed the "
                             "lower one")
        step = (hi - lo) / float(max(1, int(n_points) - 1))
        grid = [lo + step * t for t in range(int(n_points))]
    scores = [float(_gwr.gwr_criterion(y, Xr, D, b, kernel, adaptive,
                                       criterion)) for b in grid]
    return grid, scores


def gwr_calibrate(y, X, coords, kernel="gaussian", criterion="aicc",
                  adaptive=False, bounds=None, search=None, n_points=30,
                  tol=1e-4):
    """Choose a GWR bandwidth and report the fit it gives.

    Parameters
    ----------
    y, X, coords : array-like
        Response, design (include your own intercept column), and
        locations.
    kernel : {"gaussian", "bisquare", "tricube", "boxcar"}
        Gaussian weights everywhere; the other three are zero beyond the
        bandwidth, which makes each local fit use a finite neighbourhood.
    criterion : {"aicc", "cv", "aic"}
        AICc by default -- eq. 2.33, with the small-sample correction.
    adaptive : bool
        Treat the bandwidth as a neighbour count rather than a distance.
    search : {"golden", "grid"}, optional
        Defaults to golden section for a fixed bandwidth and to the grid
        for an adaptive one, which is integer-valued anyway.
    """
    y, Xr, C, n, p = _prepare(y, X, coords)
    if kernel not in _KERNELS:
        raise ValueError("gwrcal: kernel must be one of %s"
                         % (", ".join(_KERNELS)))
    if criterion not in _CRITERIA:
        raise ValueError("gwrcal: criterion must be one of %s"
                         % (", ".join(_CRITERIA)))
    if search is None:
        search = "grid" if adaptive else "golden"
    if search not in ("golden", "grid"):
        raise ValueError("gwrcal: search must be 'golden' or 'grid'")
    if adaptive and search == "golden":
        raise ValueError("gwrcal: an adaptive bandwidth is a neighbour "
                         "count, so it is searched on the integer grid; "
                         "use search='grid'")

    D = _gwr.pairwise_distances(C)
    grid, scores = bandwidth_profile(y, Xr, C, kernel, criterion,
                                     adaptive, bounds, n_points)
    if search == "grid":
        best = min(range(len(grid)), key=lambda t: scores[t])
        bw, score = grid[best], scores[best]
        if adaptive:
            bw = int(bw)
    else:
        lo, hi = _gwr._default_bounds(C) if bounds is None else bounds
        bw, score = _gwr.golden_section(
            lambda h: _gwr.gwr_criterion(y, Xr, D, h, kernel, False,
                                         criterion), lo, hi, tol=tol)
        bw, score = float(bw), float(score)

    fit = _gwr.gwr_fit(y, Xr, D, bw, kernel, adaptive)
    tr_S = float(fit["tr_S"])
    sigma2 = float(fit["sigma2"])
    resid = [float(v) for v in fit["resid"]]
    tss = sum((v - sum(y) / n) ** 2 for v in y)
    rss = sum(r * r for r in resid)
    aicc = _gwr.aicc_from_parts(n, sigma2, tr_S)
    ols_aicc = global_ols_aicc(y, Xr)

    # is the bandwidth up against the edge of the interval it was searched
    # over? then it is not a minimum, it is a boundary, and the answer is
    # "wider than anything you offered"
    edge = None
    span = grid[-1] - grid[0] if len(grid) > 1 else 0.0
    if span > 0:
        if abs(bw - grid[0]) <= 0.01 * span:
            edge = "lower"
        elif abs(bw - grid[-1]) <= 0.01 * span:
            edge = "upper"

    return RichResult(payload={
        "estimate": bw,
        "bandwidth": bw,
        "score": score,
        "criterion": criterion,
        "kernel": kernel,
        "adaptive": bool(adaptive),
        "search": search,
        "grid": grid,
        "profile": scores,
        "at_boundary": edge,
        "coefficients": [[float(v) for v in row]
                         for row in fit["params"]],
        "coefficient_se": [[float(v) for v in row]
                           for row in fit["se_params"]],
        "fitted": [float(v) for v in fit["fitted"]],
        "residuals": resid,
        "tr_S": tr_S,
        "effective_parameters": tr_S,
        "residual_df": float(fit["edf_resid"]),
        "n_rank_deficient": int(fit["n_rank_deficient"]),
        "sigma2": sigma2,
        "aicc": aicc,
        "r_squared": (1.0 - rss / tss) if tss > 0 else float("nan"),
        "ols_aicc": ols_aicc,
        "aicc_improvement": ols_aicc - aicc,
        "n": n,
        "p": p,
        "method": ("GWR bandwidth selection (Fotheringham, Brunsdon & "
                   "Charlton 2002): %s kernel, %s bandwidth, %s "
                   "minimised by %s search"
                   % (kernel, "adaptive" if adaptive else "fixed",
                      criterion.upper(), search)),
        "note": ("aicc_improvement is the global OLS AICc minus this "
                 "one; the book treats a difference of three or more as "
                 "worth having. at_boundary is set when the chosen "
                 "bandwidth sits at an end of the search interval, which "
                 "means the interval, not the data, chose it"),
    })


def gwr_bandwidth_select(y, X, coords, kernel="gaussian", **kw):
    """Optimal GWR bandwidth by CV or AIC/AICc."""
    return gwr_calibrate(y, X, coords, kernel=kernel, **kw)


def cheatsheet():
    return ("gwrcal: GWR bandwidth selection (Fotheringham, Brunsdon & "
            "Charlton 2002). Minimise CV(h) = sum (y_i - yhat_{-i})^2, "
            "AIC = 2n log sigma + n log 2pi + n + tr(S) (eq. 4.22) or "
            "AICc = 2n log sigma + n log 2pi + n (n + tr S)/(n - 2 - "
            "tr S) (eq. 2.33, the default). tr(S) is the effective "
            "number of parameters, S the hat matrix with row i equal to "
            "x(s_i)'{X'W(s_i)X}^{-1}X'W(s_i). Fixed bandwidths are "
            "distances and searched by golden section; adaptive ones are "
            "neighbour counts and searched on the integer grid. Compare "
            "aicc against ols_aicc before believing the local fit.")
