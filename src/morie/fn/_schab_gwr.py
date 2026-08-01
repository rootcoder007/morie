"""Geographically weighted regression primitives.

Shared back end for :mod:`morie.fn.spgwr`, :mod:`morie.fn.spgwrb`,
:mod:`morie.fn.spgwrk` and :mod:`morie.fn.spmsim`.

Sourcing
--------
Schabenberger & Gotway (2005) develop GWR in Sec. 6.1.3.1, pp. 316-317: the
model (6.9), the weighted-least-squares estimator, the hat matrix ``L`` whose
ith row is ``x(s_i)'{X'W(s_i)X}^{-1}X'W(s_i)``, and Cressie's residual
variance ``(Z - Z_hat)'(Z - Z_hat) / tr{(I-L)(I-L)'}``.  For the kernels
themselves the book refers back to Sec. 5.3.2 (pp. 240-241), which gives the
Epanechnikov and Gaussian *density* kernels and attributes the tri-cube to
Cleveland (1979).

The book does NOT give a bandwidth-selection criterion; it defers to
Fotheringham, Brunsdon & Charlton (2002), *Geographically Weighted
Regression: The Analysis of Spatially Varying Relationships* (Wiley), which
has now been read directly.  Everything below was checked against it:

* eq. (2.24) Gaussian ``exp[-1/2 (d_ij/b)^2]``; eq. (2.25) bi-square
  ``[1 - (d_ij/b)^2]^2 if d_ij < b, 0 otherwise`` -- note the strict
  inequality.
* eq. (2.31) ``CV = sum_i [y_i - yhat_{/=i}(b)]^2`` where the fitted value
  has "the observations for point i omitted from the calibration process".
* eq. (2.33), repeated as eq. (4.21), the AICc; eq. (4.22) the AIC.
* eq. (4.23) settles the ambiguity both the white paper and the book's own
  prose leave in "the estimated standard deviation of the error term": the
  AIC and AICc take ``sigma^2 = RSS/n``, "not that given in equation (4.7)".
* eqs. (2.17)-(2.18) ``v1 = tr(S)``, ``v2 = tr(S'S)``, with "the term
  2v1 - v2 ... can be termed the effective number of parameters".
* eq. (2.16) the residual variance for INFERENCE, denominator
  ``n - 2v1 + v2``; eqs. (2.14)-(2.15) ``Var[beta] = CC' sigma^2``.
* eq. (2.20) the hat-matrix row, identical to Schabenberger's ``L``.
* p. 60 names Golden Section search (after Greig 1980) as the optimisation.

These were corroborated against material by the same authors and their own
reference implementations:

* Charlton, M. "Geographically Weighted Regression -- White Paper"
  (University of Edinburgh GISTEAC mirror), pp. 6-8.  Gives the Gaussian
  kernel ``exp(-0.5 (d_i(u)/h)^2)``, the bisquare ``(1 - (d_i(u)/h)^2)^2``
  zero beyond ``h``, the effective number of parameters ``2 tr(S) - tr(S'S)``,
  and, citing Hurvich, Simonoff & Tsai (1998),

      AICc = 2 n log_e(sigma_hat) + n log_e(2 pi)
             + n (n + tr(S)) / (n - 2 - tr(S)).

* ``spgwr`` (Bivand & Yu, CRAN), ``R/gwr.cv.R``: ``gwr.aic.f`` fixes the
  ambiguous "estimate of the standard deviation of the residuals" as the ML
  estimate ``sigma^2 = y'(I-S)'(I-S)y / n``, and ``gwr.cv.f`` fixes the
  cross-validation score as leave-one-out -- the local fit at ``i`` is taken
  with ``w_ii`` forced to zero.  ``R/gwr.gauss.R``, ``R/gwr.bisquare.R`` and
  ``R/tricube.R`` give the three kernels in squared-distance form.

* ``GWmodel`` (CRAN), ``R/gw.weight.r`` header: the four kernels named by
  this shelf -- boxcar, gaussian, bisquare, tricube -- citing Fotheringham
  et al. (2002) pp. 56-57.  This is the only source consulted that names the
  boxcar.

* ``mgwr`` (Oshan, Li, Kang, Wolf & Fotheringham; PyPI), ``mgwr/search.py``
  ``multi_bw``: the MGWR backfitting algorithm, and ``mgwr/kernels.py`` for
  the adaptive (nearest-neighbour) bandwidth rule.

* Fotheringham, Yang & Kang (2017), Ann. Amer. Assoc. Geogr.
  107(6):1247-1265, read directly: eq. (9) SOC-RSS, eq. (10) SOC-f, the
  back-fitting algorithm of Figure 1, "we use the GWR estimates as the
  initial MGWR estimates", and SOC-f <= 1e-5 as the termination criterion.

* Fotheringham, Oshan & Li (2024), *Multiscale Geographically Weighted
  Regression: Theory and Practice*, 1st ed., CRC Press,
  doi:10.1201/9781003435464.  Sec. 2.3.2 eqs. (2.38)-(2.39) restate the
  SOC; Sec. 2.3.3.2 and Sec. 6.3 require standardization before the
  covariate-specific bandwidths can be compared.  Its eqs. (2.40)-(2.45)
  (covariate-specific hat matrices, ``ENP_k``, adjusted alpha, after Yu et
  al. 2020) are NOT implemented here.

Two things the printed material could not be transcribed as written:

* The stub docstring this module replaces printed
  ``AIC_c = 2 n log(sigma) + 2 tr(S) + 2 tr(S)' tr(S) / (n - tr(S) - 1)``.
  That is not the GWR AICc.  See :func:`aicc_from_parts` for the arithmetic
  that settles it against published output.
* Sec. 5.3.2 writes the Gaussian kernel as a *density*,
  ``(1/(lambda sqrt(2 pi))) exp(-0.5 (d/lambda)^2)``, while the GWR
  literature drops the constant.  Weighted least squares is invariant to a
  positive scalar on every weight, so both give the same fit; see
  :func:`kernel_weights` and the ``scale_invariance`` check in
  ``scripts/audit/schab_gwr_verify.py``.
"""

import numpy as np

__all__ = [
    "KERNELS",
    "adaptive_bandwidth",
    "aic_from_parts",
    "aicc_from_parts",
    "cv_score",
    "golden_section",
    "gwr_criterion",
    "gwr_fit",
    "gwr_hat_matrix",
    "kernel_weights",
    "mgwr_backfit",
    "pairwise_distances",
    "select_bandwidth",
]

KERNELS = ("gaussian", "bisquare", "tricube", "boxcar")


def pairwise_distances(coords):
    """Euclidean distance matrix for ``coords``, shape (n, d) -> (n, n)."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def kernel_weights(distance, bandwidth, kernel="gaussian", normalized=False):
    """Geographical weights ``w_i(u)`` for a GWR kernel.

    Parameters
    ----------
    distance : array-like
        Distances ``d_i(u)`` from the regression point.
    bandwidth : float
        ``h``, in the units of the coordinates.  For the truncated kernels
        it is also the support radius.
    kernel : {'gaussian', 'bisquare', 'tricube', 'boxcar'}
        ``gaussian``   ``exp(-0.5 (d/h)^2)``, positive everywhere
        ``bisquare``   ``(1 - (d/h)^2)^2`` for ``d < h``, else 0
        ``tricube``    ``(1 - (d/h)^3)^3`` for ``d < h``, else 0
        ``boxcar``     1 for ``d < h``, else 0
    normalized : bool, default False
        Return Sec. 5.3.2's Gaussian *density* form -- the same weights
        divided by ``h sqrt(2 pi)``.  Only meaningful for the Gaussian; a
        positive scalar on every weight leaves the GWR fit unchanged.

    Returns
    -------
    ndarray
        Weights, same shape as ``distance``.
    """
    d = np.asarray(distance, dtype=float)
    h = float(bandwidth)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"bandwidth must be a positive finite number, got {bandwidth!r}")
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")
    if np.any(d < 0):
        raise ValueError("distances must be non-negative")

    z = d / h
    if kernel == "gaussian":
        w = np.exp(-0.5 * z * z)
        if normalized:
            w = w / (h * np.sqrt(2.0 * np.pi))
        return w
    if normalized:
        raise ValueError("normalized= applies to the Gaussian kernel only")
    inside = z < 1.0
    if kernel == "bisquare":
        return np.where(inside, (1.0 - z * z) ** 2, 0.0)
    if kernel == "tricube":
        return np.where(inside, (1.0 - z**3) ** 3, 0.0)
    return np.where(inside, 1.0, 0.0)


def adaptive_bandwidth(distance_row, n_neighbours, eps=1.0000001):
    """Adaptive bandwidth: the ``n_neighbours``-th smallest distance.

    The rule is ``mgwr/kernels.py``: partition the distance vector and take
    the ``n_neighbours``-th order statistic, nudged by ``eps`` so that the
    neighbour itself is strictly inside a truncated kernel's support.  The
    regression point counts as its own first neighbour, matching pysal.
    """
    d = np.sort(np.asarray(distance_row, dtype=float))
    k = int(n_neighbours)
    if k < 1 or k > d.size:
        raise ValueError(f"n_neighbours must be in [1, {d.size}], got {n_neighbours!r}")
    return float(d[k - 1] * eps)


def _local_weights(d_row, bandwidth, kernel, adaptive):
    if adaptive:
        h = adaptive_bandwidth(d_row, bandwidth)
    else:
        h = float(bandwidth)
    return kernel_weights(d_row, h, kernel)


def _wls(X, y, w):
    """Weighted least squares coefficients and the operator mapping y to them.

    Solved through the SVD of the square-root-weighted design rather than by
    inverting ``X'WX``, and always through the SVD -- never by branching on
    whether an inversion raised.  A truncated kernel can leave a local fit
    with fewer non-zero weights than parameters, at which point ``X'WX`` is
    rank deficient; ``numpy.linalg.inv`` does not reliably raise there (it
    returned a garbage inverse at a condition number of 6e15, while R's
    ``solve`` raised on the same matrix and fell back to a pseudo-inverse,
    putting the two arms 1.5e+02 apart on a bisquare CV score).  Deciding
    rank explicitly, with the same cutoff in both arms, removes the
    divergence and the silent garbage together.

    Rank-deficient fits get the minimum-norm solution.  The caller is told
    how many there were rather than left to discover it.
    """
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    u, sv, vt = np.linalg.svd(Xw, full_matrices=False)
    cutoff = max(Xw.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0.0)
    keep = sv > cutoff
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        operator = np.zeros((X.shape[1], X.shape[0]))
    else:
        pinv_Xw = (vt[keep].T / sv[keep]) @ u[:, keep].T
        operator = pinv_Xw * sw[None, :]
    return operator @ y, operator, rank


def gwr_hat_matrix(X, distances, bandwidth, kernel="gaussian", adaptive=False):
    """The hat matrix ``S`` with ``y_hat = S y``.

    Row ``i`` is ``x(s_i)'{X'W(s_i)X}^{-1} X'W(s_i)`` -- the book's ``L``,
    Sec. 6.1.3.1 p. 317.
    """
    X = np.asarray(X, dtype=float)
    D = np.asarray(distances, dtype=float)
    n = X.shape[0]
    S = np.empty((n, n), dtype=float)
    for i in range(n):
        w = _local_weights(D[i], bandwidth, kernel, adaptive)
        _, operator, _ = _wls(X, np.zeros(n), w)
        S[i] = X[i] @ operator
    return S


def gwr_fit(y, X, distances, bandwidth, kernel="gaussian", adaptive=False):
    """Fit GWR at every sample location.

    Returns a dict with the local coefficients, the hat matrix and every
    quantity the selection criteria need.  ``sigma2`` is the ML estimate
    ``y'(I-S)'(I-S)y / n`` used by the AICc; ``sigma2_cressie`` is the
    book's p. 317 estimate with ``tr{(I-S)(I-S)'}`` in the denominator.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    D = np.asarray(distances, dtype=float)
    n, p = X.shape
    if y.size != n or D.shape != (n, n):
        raise ValueError("y, X and the distance matrix disagree on n")

    params = np.empty((n, p), dtype=float)
    S = np.empty((n, n), dtype=float)
    ccT = np.empty((n, p), dtype=float)
    n_deficient = 0
    for i in range(n):
        w = _local_weights(D[i], bandwidth, kernel, adaptive)
        beta_i, operator, rank = _wls(X, y, w)
        n_deficient += rank < p
        params[i] = beta_i
        S[i] = X[i] @ operator
        # eq (2.14)-(2.15): C = (X'WX)^-1 X'W, Var[beta(u_i,v_i)] = C C' sigma^2
        ccT[i] = np.einsum("ij,ij->i", operator, operator)

    fitted = S @ y
    resid = y - fitted
    ImS = np.eye(n) - S
    B = ImS.T @ ImS
    rss = float(y @ B @ y)
    tr_S = float(np.trace(S))
    tr_STS = float(np.trace(S.T @ S))
    trace_B = float(np.trace(B))
    # eq (2.16): the book's own residual variance for INFERENCE, whose
    # denominator is the effective residual degrees of freedom n - 2v1 + v2.
    # Distinct from eq (4.23)'s ML estimate, which is what the AIC and AICc
    # take (the book says so outright at eq (4.23)), and from Schabenberger's
    # Cressie estimate with tr{(I-L)(I-L)'}.  Three different denominators for
    # three different jobs; using one where another belongs shifts the answer.
    edf_resid = n - 2.0 * tr_S + tr_STS
    sigma2_gwr = rss / edf_resid if edf_resid > 0 else np.nan
    se_params = np.sqrt(np.maximum(ccT * sigma2_gwr, 0.0))
    return {
        "se_params": se_params,
        "sigma2_gwr": sigma2_gwr,
        "edf_resid": edf_resid,
        "v1": tr_S,
        "v2": tr_STS,
        "params": params,
        "fitted": fitted,
        "resid": resid,
        "S": S,
        "tr_S": tr_S,
        "tr_STS": tr_STS,
        "effective_parameters": 2.0 * tr_S - tr_STS,
        "rss": rss,
        "sigma2": rss / n,
        "sigma2_cressie": rss / trace_B if trace_B > 0 else np.nan,
        "n": n,
        "p": p,
        "bandwidth": float(bandwidth),
        "kernel": kernel,
        "adaptive": bool(adaptive),
        # Local fits with fewer estimable directions than parameters -- a
        # bandwidth too narrow for the design, not a numerical accident.
        "n_rank_deficient": int(n_deficient),
    }


def aicc_from_parts(n, sigma2, tr_S):
    """``AICc = 2n log(sigma) + n log(2 pi) + n (n + tr S) / (n - 2 - tr S)``.

    Charlton white paper p. 8, citing Hurvich et al. (1998); the same
    expression appears in ``spgwr::gwr.aic.f`` and, rearranged, in
    ``mgwr.diagnostics.get_AICc`` (Fotheringham et al. 2002 p. 61 eq. 2.33).
    """
    n = float(n)
    tr_S = float(tr_S)
    denom = n - 2.0 - tr_S
    if denom <= 0:
        return np.inf
    return 2.0 * n * np.log(np.sqrt(sigma2)) + n * np.log(2.0 * np.pi) + n * (n + tr_S) / denom


def aic_from_parts(n, sigma2, tr_S):
    """``AIC = 2n log(sigma) + n log(2 pi) + n + tr(S)``.

    Fotheringham et al. (2002) p. 96 eq. 4.22, as reported by ``spgwr``.
    """
    n = float(n)
    return 2.0 * n * np.log(np.sqrt(sigma2)) + n * np.log(2.0 * np.pi) + n + float(tr_S)


def cv_score(y, X, distances, bandwidth, kernel="gaussian", adaptive=False):
    """Leave-one-out cross-validation score ``sum_i (y_i - y_hat_{-i})^2``.

    ``spgwr::gwr.cv.f``: the local model at ``i`` is fitted with ``w_ii``
    set to zero, so ``y_i`` never contributes to its own prediction.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    D = np.asarray(distances, dtype=float)
    n = X.shape[0]
    total = 0.0
    for i in range(n):
        w = _local_weights(D[i], bandwidth, kernel, adaptive)
        w = w.copy()
        w[i] = 0.0
        if not np.any(w > 0):
            return np.inf
        beta_i, _, _ = _wls(X, y, w)
        r = y[i] - X[i] @ beta_i
        total += float(r * r)
    return total


def gwr_criterion(y, X, distances, bandwidth, kernel="gaussian", adaptive=False, criterion="cv"):
    """Objective minimised by bandwidth selection: ``'cv'``, ``'aicc'`` or ``'aic'``."""
    if criterion == "cv":
        return cv_score(y, X, distances, bandwidth, kernel, adaptive)
    if criterion not in ("aicc", "aic"):
        raise ValueError(f"unknown criterion {criterion!r}; expected 'cv', 'aicc' or 'aic'")
    fit = gwr_fit(y, X, distances, bandwidth, kernel, adaptive)
    if fit["sigma2"] <= 0:
        return np.inf
    maker = aicc_from_parts if criterion == "aicc" else aic_from_parts
    return maker(fit["n"], fit["sigma2"], fit["tr_S"])


def golden_section(func, lower, upper, tol=1e-4, max_iter=200):
    """Golden-section minimisation of a unimodal ``func`` on [lower, upper].

    ``spgwr::gwr.sel`` uses R's ``optimize`` (Brent).  Golden section is used
    here instead because it is deterministic, derivative-free and has no
    parabolic-interpolation step whose tie-breaking would have to be matched
    bit-for-bit by the R arm of this package.
    """
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lower), float(upper)
    if not b > a:
        raise ValueError("upper must exceed lower")
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = func(c), func(d)
    for _ in range(int(max_iter)):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = func(d)
    x = 0.5 * (a + b)
    return x, func(x)


def _default_bounds(coords):
    """spgwr's search interval: the bounding-box diagonal, and a 1000th of it."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    span = coords.max(axis=0) - coords.min(axis=0)
    diag = float(np.sqrt(np.sum(span**2)))
    if diag <= 0:
        raise ValueError("coordinates are degenerate; cannot set a bandwidth range")
    return diag / 1000.0, diag


def select_bandwidth(
    y,
    X,
    coords,
    kernel="gaussian",
    criterion="cv",
    adaptive=False,
    bounds=None,
    tol=1e-4,
):
    """Choose a GWR bandwidth by minimising ``criterion``.

    Returns a dict with the optimum, the objective at the optimum, the
    search interval and the criterion name.
    """
    D = pairwise_distances(coords)
    if adaptive:
        n = D.shape[0]
        lo, hi = (2.0, float(n)) if bounds is None else bounds
        grid = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1)
        scores = np.array([gwr_criterion(y, X, D, int(k), kernel, True, criterion) for k in grid])
        best = int(grid[int(np.argmin(scores))])
        return {
            "bandwidth": best,
            "score": float(scores.min()),
            "criterion": criterion,
            "bounds": (int(grid[0]), int(grid[-1])),
            "adaptive": True,
            "grid": grid.tolist(),
            "scores": scores.tolist(),
        }
    lo, hi = _default_bounds(coords) if bounds is None else bounds
    bw, score = golden_section(
        lambda h: gwr_criterion(y, X, D, h, kernel, False, criterion), lo, hi, tol=tol
    )
    return {
        "bandwidth": float(bw),
        "score": float(score),
        "criterion": criterion,
        "bounds": (float(lo), float(hi)),
        "adaptive": False,
    }


def mgwr_backfit(
    y,
    X,
    coords,
    kernel="gaussian",
    criterion="aicc",
    adaptive=False,
    tol=1e-5,
    max_iter=200,
    rss_score=False,
    bws_same_times=5,
    init_bandwidth=None,
    standardize=True,
):
    """Multiscale GWR: one bandwidth per covariate, by GAM backfitting.

    Transcribed from ``mgwr.search.multi_bw`` (the MGWR authors' own
    implementation).  Each covariate ``j`` is smoothed against the partial
    residual ``XB[:, j] + err`` by a *univariate* GWR with its own
    bandwidth; ``err`` is threaded through the inner loop so that later
    covariates in the same sweep see the earlier updates.

    ``standardize`` is on by default, which is not a stylistic choice.
    Fotheringham, Oshan & Li (2024) Sec. 2.3.3.2: "in order to effectively
    compare the values of the estimated bandwidth to each other, it is
    necessary to first standardize the input data so that y and each column
    of X have a mean of zero and variance of one before using the data in
    the MGWR calibration routine.  This normalizes the magnitude and
    dispersion of each explanatory variable so that the covariate-specific
    bandwidths can be interpreted relative to each other".  Sec. 6.3 adds
    that in the authors' own software this "is one of the default settings
    and has to be actively turned off", because "without data
    standardization, the optimized covariate-specific bandwidths will be, in
    part, a function of the variability of each covariate".  Comparing raw
    bandwidths across covariates fitted on unstandardized data compares
    quantities the source says are not comparable.  Constant columns are
    left alone -- an intercept has no variance to normalise.

    NOT implemented, and named here rather than left silently missing: the
    covariate-specific hat matrices ``R_k`` of Fotheringham, Oshan & Li
    (2024) eqs (2.40)-(2.42), the per-covariate effective parameter counts
    ``ENP_k = tr(R_k)`` of eq (2.43), and the covariate-specific adjusted
    alpha of eq (2.45), all after Yu et al. (2020).  This function returns
    bandwidths and coefficients, not MGWR inference.

    A caution the SOC makes necessary.  Both scores measure how much the
    fit MOVED, not how good it is, so a sweep that barely changes anything
    scores as converged.  When the initial single-bandwidth GWR already
    sits at the wide end of the search interval, the first sweep can leave
    every covariate there, the score is tiny, and the loop stops after two
    or three sweeps having found no scale separation at all.  Measured over
    eight seeds of a fixture with two genuinely different scales, this
    happened twice.  It is a property of the criterion, not of this port --
    the reference implementation uses the same score and the same default
    tolerance.  ``at_search_boundary`` in the return value flags it: when
    True, treat the bandwidths as a non-result and rerun from a narrower
    ``init_bandwidth``.

    Convergence uses the score of change (SOC).  ``rss_score=False`` is the
    default SOC-f,

        sqrt( (sum (XB_new - XB)^2 / n) / sum_i (sum_j XB_new[i, j])^2 ),

    and ``rss_score=True`` is SOC-RSS, ``|rss_new - rss| / rss_new``.  The
    loop stops when the score falls below ``tol``; ``bws_same_times``
    consecutive sweeps with unchanged bandwidths freeze the search.

    Returns a dict with per-covariate bandwidths, local coefficients, the
    bandwidth and score histories, and whether the tolerance was met.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    if y.shape[0] != n:
        raise ValueError("y and X disagree on n")
    D = pairwise_distances(coords)

    y_raw = y.copy()
    y_centre, y_scale = 0.0, 1.0
    x_centre = np.zeros(k)
    x_scale = np.ones(k)
    if standardize:
        y_centre = float(y.mean())
        y_scale = float(y.std(ddof=0)) or 1.0
        y = (y - y_centre) / y_scale
        for j in range(k):
            sd = float(X[:, j].std(ddof=0))
            if sd > 0:                      # leave a constant column alone
                x_centre[j] = float(X[:, j].mean())
                x_scale[j] = sd
        X = (X - x_centre) / x_scale

    def _fit(resp, design, bw):
        return gwr_fit(resp.ravel(), design, D, bw, kernel, adaptive)

    def _select(resp, design):
        return select_bandwidth(
            resp.ravel(), design, coords, kernel=kernel, criterion=criterion, adaptive=adaptive
        )["bandwidth"]

    if init_bandwidth is None:
        bw_gwr = _select(y, X)
    else:
        bw_gwr = init_bandwidth
    optim = _fit(y, X, bw_gwr)
    err = optim["resid"].reshape(-1, 1)
    XB = X * optim["params"]

    rss = float(np.sum(err**2))
    bws = np.empty(k, dtype=float)
    bw_history, score_history = [], []
    stable = 0
    converged = False
    params = np.zeros_like(X)

    for _ in range(int(max_iter)):
        new_XB = np.zeros_like(X)
        params = np.zeros_like(X)
        for j in range(k):
            temp_y = XB[:, [j]] + err
            temp_X = X[:, [j]]
            bw = bws[j] if stable >= bws_same_times else _select(temp_y, temp_X)
            sub = _fit(temp_y, temp_X, bw)
            err = sub["resid"].reshape(-1, 1)
            new_XB[:, j] = sub["fitted"]
            params[:, j] = sub["params"].ravel()
            bws[j] = bw

        if bw_history and np.all(bw_history[-1] == bws):
            stable += 1
        else:
            stable = 0

        num = float(np.sum((new_XB - XB) ** 2)) / n
        den = float(np.sum(np.sum(new_XB, axis=1) ** 2))
        score = np.sqrt(num / den) if den > 0 else np.inf
        XB = new_XB

        if rss_score:
            predy = np.sum(params * X, axis=1).reshape(-1, 1)
            new_rss = float(np.sum((y - predy) ** 2))
            score = abs((new_rss - rss) / new_rss) if new_rss > 0 else 0.0
            rss = new_rss

        score_history.append(float(score))
        bw_history.append(bws.copy())
        if score < tol:
            converged = True
            break

    fitted_std = np.sum(params * X, axis=1)
    # Back to the original units, so `fitted`/`resid`/`rss` mean what their
    # names say regardless of `standardize`.  The coefficients stay on the
    # standardized scale -- that is the scale on which the source says they
    # are comparable across covariates -- with the centres and scales
    # returned so a caller can undo it.
    fitted = fitted_std * y_scale + y_centre
    # Diagnostic for the false convergence described above: every covariate
    # left at the top of its search interval is a non-result, not a finding.
    at_boundary = (False if adaptive
                   else bool(np.all(bws > 0.95 * _default_bounds(coords)[1])))
    return {
        "bandwidths": bws.copy(),
        "at_search_boundary": at_boundary,
        "standardized": bool(standardize),
        "y_centre": y_centre,
        "y_scale": y_scale,
        "x_centre": x_centre,
        "x_scale": x_scale,
        "params": params,
        "fitted": fitted,
        "resid": y_raw.ravel() - fitted,
        "bandwidth_gwr": bw_gwr,
        "bandwidth_history": [b.tolist() for b in bw_history],
        "score_history": score_history,
        "n_iter": len(score_history),
        "converged": converged,
        "criterion": criterion,
        "kernel": kernel,
    }
