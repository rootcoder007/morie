# morie.fn -- shared engine (rootcoder007/morie)
"""Semivariogram estimation, fitting and kriging diagnostics.

Every formula is taken from Schabenberger and Gotway (2005),
*Statistical Methods for Spatial Data Analysis*, Chapman and Hall/CRC,
and the equation numbers below are the book's own. Page numbers refer
to the printed pages.

The estimators in section 4.4 differ in how they summarise the squared
differences, and the difference is not cosmetic: the Matheron estimator
is unbiased but has an unbounded influence function, so a single
outlying observation contaminates the estimate at every lag it
participates in. Example 4.3 (p. 157-161) is reproduced in the test
References
----------
Schabenberger, O. & Gotway, C. A. (2005) *Statistical Methods for
Spatial Data Analysis*, Texts in Statistical Science, Chapman &
Hall/CRC, Boca Raton, ISBN 1-58488-322-7.
Sec. 4.4, eqs (4.24) and (4.34), and Example 4.3 (pp. 157-161).

Matheron, G. (1962) *Traite de Geostatistique Appliquee, Tome I*,
Memoires du Bureau de Recherches Geologiques et Minieres no. 14,
Editions Technip, Paris -- where Schabenberger & Gotway (p. 153)
attribute the classical estimator. Print monograph; not digitised.

Matheron, G. (1963) "Principles of geostatistics", *Economic Geology*
58(8), 1246-1266, doi:10.2113/gsecongeo.58.8.1246 -- the English
exposition that is usually cited in its place. Matheron's *The Theory
of Regionalized Variables and Its Applications* (Les Cahiers du Centre
de Morphologie Mathematique de Fontainebleau) is freely available from
the Fontainebleau library and is in fetched-wave3.

Cressie, N. & Hawkins, D. M. (1980) "Robust estimation of the
variogram, I", *Journal of the International Association for
Mathematical Geology* 12, 115-125 -- the robust alternative to the
classical estimator, and the reason the influence-function contrast
above matters.

Cressie, N. (1985) "Fitting variogram models by weighted least
squares", *Journal of the International Association for Mathematical
Geology* 17, 563-586 -- eq (4.34).

Zimmerman, D. L. & Zimmerman, M. B. (1991) "A comparison of spatial
semivariogram estimators and corresponding kriging predictors",
*Technometrics* 33, 77-91 -- the OLS/WLS comparison quoted at p. 165.

suite precisely because it shows this happening.
"""

from . import _array_core as np

__all__ = [
    "pair_differences",
    "lag_bins",
    "matheron",
    "cressie_hawkins",
    "variogram_model",
    "wls_weights",
    "fit_variogram_wls",
    "composite_likelihood_fit",
    "gaussian_neg2loglik",
    "reml_neg2loglik",
    "MODELS",
]

MODELS = ("exponential", "spherical", "gaussian", "linear")


def pair_differences(coords, z):
    """All distinct pairs: separation distance and value difference.

    Returns ``(h, diff)`` over the ``n(n-1)/2`` unique pairs, which is
    the semivariogram cloud of section 4.4.1.
    """
    P = np.atleast_2d(np.asarray(coords, dtype=float))
    if P.shape[0] == 1 and P.shape[1] != np.asarray(z).size:
        P = P.T
    zz = np.asarray(z, dtype=float).ravel()
    n = zz.size
    if P.shape[0] != n:
        P = P.T
    if P.shape[0] != n:
        raise ValueError(
            "coords has %d rows for %d values." % (P.shape[0], n)
        )
    if n < 2:
        raise ValueError("need at least 2 locations, got %d." % n)
    i, j = np.triu_indices(n, k=1)
    h = np.sqrt(np.sum((P[i] - P[j]) ** 2, axis=1))
    return h, zz[i] - zz[j]


def lag_bins(h, bins=None, cutoff=None, tol=None):
    """Group separations into lag classes.

    The book recommends at least 30, preferably 50, pairs per class
    (p. 153) and computing the empirical semivariogram only to about
    half the maximum separation (p. 155), because the number of
    available pairs collapses at long range while the semivariance
    itself keeps rising -- equation (4.25) makes that variance
    explosion explicit.

    Passing an explicit sequence of bin EDGES uses them as given; that
    is what reproduces a worked example whose lag classes are dictated
    by the data rather than by a rule.
    """
    h = np.asarray(h, dtype=float)
    if bins is not None and np.ndim(bins) > 0:
        edges = np.asarray(bins, dtype=float)
    else:
        k = 15 if bins is None else int(bins)
        if k < 1:
            raise ValueError("need at least 1 lag class, got %d." % k)
        top = float(np.max(h)) / 2.0 if cutoff is None else float(cutoff)
        if top <= 0:
            raise ValueError("cutoff must be positive, got %r." % cutoff)
        edges = np.linspace(0.0, top, k + 1)
    idx = np.digitize(h, edges[1:-1], right=True)
    inside = (h > edges[0] - 1e-12) & (h <= edges[-1] + 1e-12)
    return edges, idx, inside


def _grouped(h, diff, bins, cutoff, exact):
    """Shared binning for both estimators.

    ``exact=True`` treats every distinct separation as its own lag,
    which is what a small worked example needs.
    """
    if exact:
        uniq = np.unique(np.round(h, 12))
        groups = [np.isclose(h, u) for u in uniq]
        return uniq, groups
    edges, idx, inside = lag_bins(h, bins, cutoff)
    centres, groups = [], []
    for b in range(len(edges) - 1):
        m = inside & (idx == b)
        if m.any():
            centres.append(float(np.mean(h[m])))
            groups.append(m)
    return np.asarray(centres), groups


def matheron(coords, z, bins=None, cutoff=None, exact=False):
    r"""Matheron's classical estimator, equation (4.24), p. 153.

    .. math::
       \hat\gamma(h) = \frac{1}{2|N(h)|}
                       \sum_{N(h)} \{Z(s_i) - Z(s_j)\}^2

    Unbiased, even, and zero at zero lag. Its approximate variance is
    equation (4.25),
    :math:`\mathrm{Var}[\hat\gamma(h_i)] \approx
    2\gamma(h_i)^2 / |N(h_i)|`, which is what the weighted least
    squares fit later uses as its weight.
    """
    h, d = pair_differences(coords, z)
    centres, groups = _grouped(h, d, bins, cutoff, exact)
    gam = np.array([float(np.mean(d[m] ** 2) / 2.0) for m in groups])
    npair = np.array([int(m.sum()) for m in groups])
    var = np.where(npair > 0, 2.0 * gam ** 2 / np.maximum(npair, 1), np.nan)
    return centres, gam, npair, var


def cressie_hawkins(coords, z, bins=None, cutoff=None, exact=False):
    r"""The robust estimator, equation (4.26), p. 160.

    .. math::
       \hat\gamma_{CH}(h) = \frac{1}{2}
       \left\{\frac{1}{|N(h)|}\sum_{N(h)}
              |Z(s_i) - Z(s_j)|^{1/2}\right\}^{4}
       \Big/ \left(0.457 + \frac{0.494}{|N(h)|}\right)

    Averaging the square-root differences BEFORE raising to the fourth
    power is what limits an outlier's leverage; the denominator
    restores approximate unbiasedness.

    The book derives the bias correction as
    :math:`0.457 + 0.494/|N(h)| + 0.045/|N(h)|^2` and then drops the
    last term when writing equation (4.26), noting it "contributes very
    little ... particularly if |N(h)| is large". Equation (4.26) as
    printed is implemented here, because that is the estimator the
    book's own worked Example 4.3 evaluates -- its factor 0.704 at
    :math:`|N(h)| = 2` is :math:`0.457 + 0.494/2` exactly, with no
    :math:`0.045/4` term. ``full_correction=True`` is available on the
    public wrapper for the three-term version.

    Robust here means resistant to slight contamination of a Gaussian
    field, not to gross contamination: the influence function is still
    unbounded and the breakdown point is still zero (p. 161).
    """
    h, d = pair_differences(coords, z)
    centres, groups = _grouped(h, d, bins, cutoff, exact)
    gam, npair = [], []
    for m in groups:
        nh = int(m.sum())
        root = float(np.mean(np.sqrt(np.abs(d[m]))))
        gam.append(0.5 * root ** 4 / (0.457 + 0.494 / nh))
        npair.append(nh)
    return centres, np.asarray(gam), np.asarray(npair, dtype=int)


def variogram_model(h, model, nugget, psill, rng):
    """Isotropic semivariogram models of section 4.3.

    The exponential and Gaussian forms use the PRACTICAL range
    convention -- the distance at which the correlation has decayed to
    0.05 -- which is why the 3 appears in the exponent. Mixing that up
    with the scale parameter rescales every fitted range by 3.
    """
    if model not in MODELS:
        raise ValueError("model must be one of %s, got %r." % (MODELS, model))
    h = np.asarray(h, dtype=float)
    a = max(float(rng), 1e-12)
    if model == "exponential":
        g = 1.0 - np.exp(-3.0 * h / a)
    elif model == "gaussian":
        g = 1.0 - np.exp(-3.0 * (h / a) ** 2)
    elif model == "spherical":
        t = np.clip(h / a, 0.0, 1.0)
        g = 1.5 * t - 0.5 * t ** 3
    else:
        g = np.minimum(h / a, 1.0)
    return np.where(h <= 0, 0.0, nugget + psill * g)


def wls_weights(npair, gamma_model):
    r"""Cressie's weights, equation (4.34), p. 165.

    Minimising
    :math:`\sum_m |N(h_m)| \{\hat\gamma(h_m) -
    \gamma(h_m,\theta)\}^2 / (2\gamma(h_m,\theta)^2)`
    is weighted least squares with the reciprocal of the approximate
    variance (4.33). The weights depend on the parameters, so the fit
    has to be iteratively re-weighted; the book is explicit that the
    off-diagonal entries of the true covariance are appreciable, so
    this is an approximation and not the generalised criterion (4.31).
    """
    g = np.maximum(np.asarray(gamma_model, dtype=float), 1e-12)
    return np.asarray(npair, dtype=float) / (2.0 * g ** 2)


def _start(centres, gam):
    sill = float(np.nanmax(gam)) if gam.size else 1.0
    return np.array([
        max(float(np.nanmin(gam)), 1e-8),
        max(sill - float(np.nanmin(gam)), 1e-8),
        max(float(np.nanmax(centres)) / 2.0, 1e-8),
    ])


def _nelder_mead(fn, x0, max_iter=2000, tol=1e-12):
    """Nelder-Mead on the log scale, so the parameters stay positive.

    Written out rather than imported: the fit has to behave identically
    in the R port, and a shared derivative-free simplex is easier to
    keep in step than two different optimiser implementations.
    """
    n = x0.size
    sim = np.vstack([x0] + [x0 + np.eye(n)[i] * 0.35 for i in range(n)])
    f = np.array([fn(p) for p in sim])
    for _ in range(int(max_iter)):
        order = np.argsort(f)
        sim, f = sim[order], f[order]
        if abs(f[-1] - f[0]) <= tol * (abs(f[0]) + tol):
            break
        cen = sim[:-1].mean(axis=0)
        xr = cen + (cen - sim[-1])
        fr = fn(xr)
        if fr < f[0]:
            xe = cen + 2.0 * (cen - sim[-1])
            fe = fn(xe)
            sim[-1], f[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < f[-2]:
            sim[-1], f[-1] = xr, fr
        else:
            xc = cen + 0.5 * (sim[-1] - cen)
            fc = fn(xc)
            if fc < f[-1]:
                sim[-1], f[-1] = xc, fc
            else:
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                f[1:] = np.array([fn(p) for p in sim[1:]])
    order = np.argsort(f)
    return sim[order][0], float(f[order][0])


def fit_variogram_wls(centres, gam, npair, model="exponential",
                      weights="cressie", max_reweight=50, tol=1e-10):
    """Fit a semivariogram model by (iteratively re-weighted) WLS.

    ``weights='ols'`` is the :math:`R = \\phi I` simplification of
    p. 165. Zimmerman and Zimmerman (1991) found OLS and WLS perform
    about equally; the book's point is that the real efficiency loss
    comes from ignoring the CORRELATIONS among the empirical
    semivariogram values, which neither addresses.
    """
    centres = np.asarray(centres, dtype=float)
    gam = np.asarray(gam, dtype=float)
    npair = np.asarray(npair, dtype=float)
    ok = np.isfinite(gam) & (npair > 0)
    centres, gam, npair = centres[ok], gam[ok], npair[ok]
    if centres.size < 3:
        raise ValueError(
            "need at least 3 usable lag classes to fit 3 parameters, got %d."
            % centres.size
        )
    theta = _start(centres, gam)
    w = np.ones_like(gam) if weights == "ols" else None
    prev = None
    for it in range(int(max_reweight)):
        ww = w if w is not None else wls_weights(
            npair, variogram_model(centres, model, *theta)
        )

        def obj(p, ww=ww):
            t = np.exp(p)
            r = gam - variogram_model(centres, model, t[0], t[1], t[2])
            return float(np.sum(ww * r ** 2))

        best, _ = _nelder_mead(obj, np.log(np.maximum(theta, 1e-10)))
        theta = np.exp(best)
        if weights == "ols":
            break
        if prev is not None and np.max(np.abs(theta - prev)) < tol:
            break
        prev = theta.copy()
    fitted = variogram_model(centres, model, *theta)
    ww = np.ones_like(gam) if weights == "ols" else wls_weights(npair, fitted)
    return {
        "nugget": float(theta[0]),
        "psill": float(theta[1]),
        "range": float(theta[2]),
        "sill": float(theta[0] + theta[1]),
        "model": model,
        "weights": weights,
        "fitted": fitted,
        "objective": float(np.sum(ww * (gam - fitted) ** 2)),
        "iterations": it + 1,
    }


def composite_likelihood_fit(coords, z, model="exponential",
                             max_iter=60, tol=1e-10):
    r"""Composite likelihood, equation (4.44), p. 171.

    The estimating equation is

    .. math::
       CS(\theta; T^{(2)}) = 2\sum_{i<j}
         \frac{\partial \gamma(h_{ij},\theta)}{\partial\theta}
         \frac{1}{8\gamma(h_{ij},\theta)^2}
         \left\{T^{(3)}_{ij} - 2\gamma(h_{ij},\theta)\right\} = 0,

    which is the generalised estimating equation (4.43) weighted by
    :math:`1/(8\gamma^2)`. That weight is not a tuning choice: under
    the Gaussian assumption
    :math:`T^{(3)}_{ij}/2\gamma(h_{ij},\theta) \sim \chi^2_1`, so
    :math:`\mathrm{Var}[T^{(3)}_{ij}] = 8\gamma(h_{ij},\theta)^2`
    exactly, and the composite likelihood is the variance-weighted
    version of the GEE.

    Crucially this fits the semivariogram CLOUD, pair by pair, not the
    binned empirical semivariogram -- so no lag classes are chosen and
    no binning decision influences the answer.
    """
    h, d = pair_differences(coords, z)
    t3 = d ** 2
    theta = np.array([
        max(0.1 * float(np.var(z)), 1e-8),
        max(0.9 * float(np.var(z)), 1e-8),
        max(float(np.max(h)) / 3.0, 1e-8),
    ])

    # Equation (4.44) is a SCORE equation, and the book's instruction is
    # to solve it "by (nonlinear) weighted least squares ... with a
    # Gauss-Newton algorithm" -- which holds the weight fixed within each
    # pass and updates it between passes.
    #
    # Minimising sum (T3 - 2 gamma)^2 / (8 gamma^2) directly instead is a
    # different and badly behaved problem: the weight is part of what is
    # being optimised, so the fit can shrink its own weight by inflating
    # gamma. As gamma grows the summand tends to 4 gamma^2 / 8 gamma^2 =
    # 1/2, so the objective approaches a finite ceiling and imposes no
    # penalty at all on a runaway sill. Measured on a field simulated
    # with sill 2.0, the direct version returned 6.09 while the
    # iteratively re-weighted one returns a value in line with WLS.
    prev = None
    for it in range(int(max_iter)):
        g_cur = np.maximum(
            variogram_model(h, model, theta[0], theta[1], theta[2]), 1e-12
        )
        w = 1.0 / (8.0 * g_cur ** 2)

        def obj(p, w=w):
            t = np.exp(p)
            g = variogram_model(h, model, t[0], t[1], t[2])
            return float(np.sum(w * (t3 - 2.0 * g) ** 2))

        best, val = _nelder_mead(obj, np.log(np.maximum(theta, 1e-10)))
        theta = np.exp(best)
        if prev is not None and np.max(np.abs(theta - prev)) < tol:
            break
        prev = theta.copy()

    g_fin = np.maximum(
        variogram_model(h, model, theta[0], theta[1], theta[2]), 1e-12
    )
    val = float(np.sum((t3 - 2.0 * g_fin) ** 2 / (8.0 * g_fin ** 2)))
    # A bounded model cannot fit an unbounded variogram. Under a linear
    # trend the semivariance keeps climbing (equation 5.35), and the fit
    # answers by pushing the range towards infinity -- observed running
    # to 1e11 on a pure-trend design. That is a diagnosis, not a fit, so
    # it is reported rather than returned as though it were a sill.
    hmax = float(np.max(h))
    diverged = bool(theta[2] > 10.0 * hmax)
    return {
        "nugget": float(theta[0]),
        "psill": float(theta[1]),
        "range": float(theta[2]),
        "sill": float(theta[0] + theta[1]),
        "model": model,
        "objective": val,
        "iterations": it + 1,
        "n_pairs": int(h.size),
        "converged": not diverged,
        "diverged_note": (
            None if not diverged else
            "the fitted range (%.3g) exceeds ten times the largest "
            "separation in the data (%.3g), which means no bounded sill was "
            "found; the usual cause is a trend in the mean, whose squared "
            "difference is added to the semivariance by equation (5.35). "
            "Detrend first, or fit the 'linear' model."
            % (theta[2], hmax)
        ),
    }


def _cov_matrix(coords, model, nugget, psill, rng):
    P = np.atleast_2d(np.asarray(coords, dtype=float))
    D = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    sill = nugget + psill
    C = sill - variogram_model(D, model, nugget, psill, rng)
    return C, D


def gaussian_neg2loglik(coords, z, model, nugget, psill, rng, X=None):
    r"""Minus twice the Gaussian log likelihood, equation (4.35), p. 166.

    .. math::
       \varphi(\mu;\theta) = \ln|\Sigma(\theta)| + n\ln 2\pi
         + (Z - 1\mu)'\Sigma(\theta)^{-1}(Z - 1\mu),

    with :math:`\mu` profiled out by the generalised least squares
    estimator (4.36). Maximum likelihood makes no allowance for the
    degrees of freedom spent on the mean, so its covariance estimates
    are biased downward -- for independent data the bias is exactly
    :math:`-\theta/n` (p. 167).
    """
    zz = np.asarray(z, dtype=float).ravel()
    n = zz.size
    C, _ = _cov_matrix(coords, model, nugget, psill, rng)
    C = C + np.eye(n) * 1e-10 * max(float(np.trace(C)) / n, 1e-12)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return np.inf, np.nan
    Xd = np.ones((n, 1)) if X is None else np.atleast_2d(
        np.asarray(X, dtype=float)
    )
    if Xd.shape[0] != n:
        Xd = Xd.T
    Ci_X = np.linalg.solve(C, Xd)
    beta = np.linalg.solve(Xd.T @ Ci_X, Ci_X.T @ zz)
    r = zz - Xd @ beta
    quad = float(r @ np.linalg.solve(C, r))
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return logdet + n * np.log(2.0 * np.pi) + quad, beta


def reml_neg2loglik(coords, z, model, nugget, psill, rng, X=None):
    r"""Minus twice the REML log likelihood, p. 263.

    .. math::
       \varphi_R(\theta) = \ln|\Sigma(\theta)|
         + \ln|X'\Sigma(\theta)^{-1}X|
         + r'\Sigma(\theta)^{-1}r + (n-k)\ln 2\pi.

    The extra :math:`\ln|X'\Sigma^{-1}X|` term is the whole point: it
    accounts for the degrees of freedom spent estimating the mean, and
    is what removes the downward bias of the ML variance estimate.

    Because the two objectives are likelihoods of DIFFERENT data --
    :math:`Z` for ML and the error contrasts :math:`KZ` for REML -- a
    likelihood-ratio comparison of REML fits is meaningful only when
    the mean structures match (p. 168).
    """
    zz = np.asarray(z, dtype=float).ravel()
    n = zz.size
    C, _ = _cov_matrix(coords, model, nugget, psill, rng)
    C = C + np.eye(n) * 1e-10 * max(float(np.trace(C)) / n, 1e-12)
    Xd = np.ones((n, 1)) if X is None else np.atleast_2d(
        np.asarray(X, dtype=float)
    )
    if Xd.shape[0] != n:
        Xd = Xd.T
    k = Xd.shape[1]
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return np.inf, np.nan
    Ci_X = np.linalg.solve(C, Xd)
    XtCiX = Xd.T @ Ci_X
    sign, logdet_xcx = np.linalg.slogdet(XtCiX)
    if sign <= 0:
        return np.inf, np.nan
    beta = np.linalg.solve(XtCiX, Ci_X.T @ zz)
    r = zz - Xd @ beta
    quad = float(r @ np.linalg.solve(C, r))
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return (logdet + logdet_xcx + quad + (n - k) * np.log(2.0 * np.pi),
            beta)


def cheatsheet():
    return (
        "_schaben: Matheron and Cressie-Hawkins semivariograms, WLS / ML / "
        "REML / composite-likelihood fitting, all against Schabenberger and "
        "Gotway's own equation numbers"
    )
