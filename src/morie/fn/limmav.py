r"""voom: precision weights for RNA-seq log-counts.

Law, C. W., Chen, Y., Shi, W., & Smyth, G. K. (2014) "voom: precision weights
unlock linear model analysis tools for RNA-seq read counts", *Genome Biology*
15:R29.

Counts are heteroscedastic, so normal linear models do not apply to them
directly. voom's move is to transform once and then *carry the variance
structure along as weights*, which lets the whole normal-theory toolkit run
on RNA-seq.

The log-counts per million for count :math:`r_{gi}` in a library of size
:math:`R_i` is

.. math:: y_{gi} = \log_2\left(\frac{r_{gi} + 0.5}{R_i + 1.0}
          \times 10^6\right),

where "the counts are offset away from zero by 0.5 to avoid taking the log of
zero, and to reduce the variability of log-cpm for low expression genes. The
library size is offset by 1 to ensure that :math:`(r_{gi}+0.5)/(R_i+1)` is
strictly less than 1 as well as strictly greater than zero."

Then, in the paper's own order:

1. fit the linear model to :math:`y_g` by **ordinary** least squares, giving
   :math:`\hat\beta_g`, fitted values :math:`\hat\mu_{gi}` and residual
   standard deviations :math:`s_g`;
2. convert each gene's average log-cpm to an average log-count,
   :math:`\tilde r = \bar y_g + \log_2 \tilde R - \log_2 10^6`, with
   :math:`\tilde R` "the geometric mean of the library sizes plus one";
3. fit a LOWESS curve to :math:`s_g^{1/2}` against :math:`\tilde r` --
   "square-root standard deviations are used because they are roughly
   symmetrically distributed" -- and read it as a piecewise linear function
   :math:`lo()` by interpolating between the ordered :math:`\tilde r`;
4. convert each *fitted* log-cpm to a fitted log-count,
   :math:`\hat\lambda_{gi} = \hat\mu_{gi} + \log_2(R_i + 1) - \log_2 10^6`;
5. the precision weight is
   :math:`w_{gi} = lo(\hat\lambda_{gi})^{-4}` -- the inverse *variance*,
   since :math:`lo` predicts a square-root standard deviation.

The weights are per observation, not per gene, which is the point: "different
samples may be sequenced to different depths, so different count sizes may be
quite different even if the cpm values are the same".

What follows the weighting is limma's own pipeline:

    Smyth, G. K. (2004) "Linear models and empirical Bayes methods for
    assessing differential expression in microarray experiments",
    *Statistical Applications in Genetics and Molecular Biology* 3(1),
    Article 3.

The gene-wise variances are moderated toward a prior fitted across all
genes. With prior information equivalent to an estimator :math:`s_0^2` on
:math:`d_0` degrees of freedom, the posterior mean of :math:`\sigma_g^{-2}`
gives

.. math:: \tilde s_g^2 = \frac{d_0 s_0^2 + d_g s_g^2}{d_0 + d_g},
          \qquad
          \tilde t_{gj} = \frac{\hat\beta_{gj}}{\tilde s_g \sqrt{v_{gj}}},

and the moderated statistic is t-distributed on :math:`d_g + d_0` degrees of
freedom -- "the added degrees of freedom ... reflect the extra information
which is borrowed ... from the ensemble of genes". The two ends of the
spectrum are worth stating because they bracket what moderation does: "the
moderated t reduces to the ordinary t-statistic if :math:`d_0 = 0` and at the
opposite end of the spectrum is proportional to the coefficient
:math:`\hat\beta_{gj}` if :math:`d_0 = \infty`."

The hyperparameters are estimated by the paper's closed forms, matching the
first two moments of :math:`\log s_g^2` -- chosen because "the moments of
:math:`\log s_g^2` are finite for any degrees of freedom and because the
distribution ... is more nearly normal". With
:math:`e_g = \log s_g^2 - \psi(d_g/2) + \log(d_g/2)`,

.. math:: \psi'(d_0/2) = \mathrm{mean}\{(e_g - \bar e)^2 G/(G-1)
          - \psi'(d_g/2)\}, \qquad
          s_0^2 = \exp\{\bar e + \psi(d_0/2) - \log(d_0/2)\},

solved for :math:`d_0` by the monotone Newton iteration of the paper's
appendix. When that mean is non-positive "there is no evidence that the
underlying variances vary between genes", so :math:`d_0 = \infty` and
:math:`s_0^2 = \exp(\bar e)` -- every gene gets the same variance.
``moderate=False`` returns the ordinary weighted-least-squares t instead.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

from .deseq2 import benjamini_hochberg

__all__ = ["limmav", "voom", "limma_voom", "log_cpm", "lowess",
           "voom_weights", "weighted_lm", "ebayes", "digamma",
           "trigamma", "trigamma_inverse"]


def log_cpm(counts, lib_sizes=None, prior_count=0.5, lib_offset=1.0):
    r"""The paper's log-cpm:
    :math:`\log_2((r + 0.5)/(R + 1) \times 10^6)`."""
    K = [[float(v) for v in row] for row in counts]
    if not K or not K[0]:
        raise ValueError("limmav: counts must be a non-empty gene x sample "
                         "matrix")
    m = len(K[0])
    for row in K:
        if len(row) != m:
            raise ValueError("limmav: ragged count matrix")
        if any(v < 0 for v in row):
            raise ValueError("limmav: counts must be non-negative")
    if lib_sizes is None:
        R = [sum(row[j] for row in K) for j in range(m)]
    else:
        R = [float(v) for v in lib_sizes]
        if len(R) != m:
            raise ValueError("limmav: one library size per sample")
    if any(v <= 0 for v in R):
        raise ValueError("limmav: library sizes must be positive")
    y = [[math.log((row[j] + prior_count) / (R[j] + lib_offset) * 1e6, 2)
          for j in range(m)] for row in K]
    return y, R


def lowess(x, y, span=0.5, iterations=3):
    r"""Robust locally weighted regression (Cleveland's LOWESS).

    Tricube neighbour weights over a span of the data, local linear fit,
    then ``iterations`` robustness passes reweighting by the bisquare of
    the scaled residuals -- which is what makes the trend "statistically
    robust" and able to "provide a trend line through the majority of the
    standard deviations" rather than chasing outliers.

    Returns the fitted values at ``x``.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("limmav: x and y must have the same length")
    if n == 0:
        raise ValueError("limmav: nothing to smooth")
    if not 0.0 < span <= 1.0:
        raise ValueError("limmav: span must lie in (0, 1]")
    order = sorted(range(n), key=lambda i: x[i])
    xs = [x[i] for i in order]
    ys = [y[i] for i in order]
    q = max(2, int(math.ceil(span * n)))
    rw = [1.0] * n
    fitted = list(ys)
    for it in range(int(iterations) + 1):
        for i in range(n):
            lo = max(0, min(i - q // 2, n - q))
            hi = lo + q
            d = max(abs(xs[i] - xs[lo]), abs(xs[hi - 1] - xs[i]), 1e-12)
            sw = sx = sy = sxx = sxy = 0.0
            for k in range(lo, hi):
                u = abs(xs[k] - xs[i]) / d
                w = (1.0 - u ** 3) ** 3 if u < 1.0 else 0.0
                w *= rw[k]
                if w <= 0:
                    continue
                sw += w
                sx += w * xs[k]
                sy += w * ys[k]
                sxx += w * xs[k] * xs[k]
                sxy += w * xs[k] * ys[k]
            if sw <= 0:
                fitted[i] = ys[i]
                continue
            den = sw * sxx - sx * sx
            if abs(den) < 1e-12:
                fitted[i] = sy / sw
            else:
                b = (sw * sxy - sx * sy) / den
                a = (sy - b * sx) / sw
                fitted[i] = a + b * xs[i]
        if it == int(iterations):
            break
        res = [abs(ys[i] - fitted[i]) for i in range(n)]
        s = sorted(res)[n // 2]
        if s <= 0:
            break
        rw = [(1.0 - min(r / (6.0 * s), 1.0) ** 2) ** 2 for r in res]
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = fitted[pos]
    return out


def digamma(x):
    r""":math:`\psi(x)`, by recurrence to a large argument then the
    asymptotic series."""
    x = float(x)
    if x <= 0.0:
        raise ValueError("limmav: digamma needs x > 0")
    tot = 0.0
    while x < 10.0:
        tot -= 1.0 / x
        x += 1.0
    inv2 = 1.0 / (x * x)
    return (tot + math.log(x) - 0.5 / x -
            inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0)))


def trigamma(x):
    r""":math:`\psi'(x)`."""
    x = float(x)
    if x <= 0.0:
        raise ValueError("limmav: trigamma needs x > 0")
    tot = 0.0
    while x < 20.0:
        tot += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return tot + inv * (1.0 + 0.5 * inv + inv2 * (
        1.0 / 6.0 + inv2 * (-1.0 / 30.0 + inv2 * (
            1.0 / 42.0 - inv2 / 30.0))))


def _tetragamma(x):
    r""":math:`\psi''(x)`, needed by the appendix's Newton step."""
    x = float(x)
    tot = 0.0
    while x < 20.0:
        tot -= 2.0 / (x * x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return tot - inv2 * (1.0 + inv * (1.0 + inv2 * (
        1.0 / 6.0 - inv2 * (1.0 / 6.0 - 3.0 * inv2 / 10.0))))


def trigamma_inverse(x, tol=1e-8, max_iter=60):
    r"""Solve :math:`\psi'(y) = x`, by the paper's appendix.

    "Set :math:`y_0 = 0.5 + 1/x`. Then iterate
    :math:`y_{i+1} = y_i + \delta_i` with
    :math:`\delta_i = \psi'(y_i)\{1 - \psi'(y_i)/x\}/\psi''(y_i)`",
    which is monotonically convergent because :math:`f = 1/\psi'` is convex
    with :math:`f(y_0) \ge z`. The paper's overflow guards are kept:
    :math:`y = 1/\sqrt{x}` above :math:`10^7` and :math:`y = 1/x` below
    :math:`10^{-6}`.
    """
    x = float(x)
    if x <= 0.0:
        raise ValueError("limmav: trigamma_inverse needs x > 0")
    if x > 1e7:
        return 1.0 / math.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(int(max_iter)):
        tri = trigamma(y)
        tet = _tetragamma(y)
        if tet == 0.0:
            break
        d = tri * (1.0 - tri / x) / tet
        y += d
        if -d / y < tol:
            break
    return y


def ebayes(sigma2, df, robust_floor=1e-12):
    r"""Smyth (2004) section 6.2: estimate :math:`d_0` and :math:`s_0^2`
    and moderate the variances.

    ``sigma2`` are the gene-wise residual variances :math:`s_g^2` and ``df``
    their degrees of freedom :math:`d_g` (a scalar or one per gene).

    Returns ``{"d0", "s0_sq", "s2_post", "df_total", "no_gene_variation"}``
    where ``s2_post`` is :math:`\tilde s_g^2` and ``df_total`` is
    :math:`d_g + d_0`.
    """
    s2 = [float(v) for v in sigma2]
    G = len(s2)
    if G == 0:
        raise ValueError("limmav: no variances to moderate")
    dg = [float(df)] * G if not isinstance(df, (list, tuple)) else \
        [float(v) for v in df]
    if len(dg) != G:
        raise ValueError("limmav: one degrees-of-freedom value per gene")
    use = [g for g in range(G) if s2[g] > robust_floor and dg[g] > 0]
    if not use:
        raise ValueError("limmav: every gene has zero variance or zero "
                         "degrees of freedom")
    e = [math.log(s2[g]) - digamma(dg[g] / 2.0) + math.log(dg[g] / 2.0)
         for g in use]
    ebar = sum(e) / len(e)
    n = len(e)
    if n > 1:
        target = sum((v - ebar) ** 2 for v in e) * n / (n - 1.0) / n
    else:
        target = 0.0
    target -= sum(trigamma(dg[g] / 2.0) for g in use) / len(use)
    if target <= 0.0:
        # "there is no evidence that the underlying variances vary between
        # genes so d0 is set to positive infinity and s0^2 = exp(ebar)"
        d0 = float("inf")
        s0_sq = math.exp(ebar)
        post = [s0_sq] * G
        return {"d0": d0, "s0_sq": s0_sq, "s2_post": post,
                "df_total": [float("inf")] * G,
                "no_gene_variation": True}
    d0 = 2.0 * trigamma_inverse(target)
    s0_sq = math.exp(ebar + digamma(d0 / 2.0) - math.log(d0 / 2.0))
    post = [(d0 * s0_sq + dg[g] * s2[g]) / (d0 + dg[g])
            if dg[g] > 0 else s0_sq for g in range(G)]
    return {"d0": d0, "s0_sq": s0_sq, "s2_post": post,
            "df_total": [dg[g] + d0 for g in range(G)],
            "no_gene_variation": False}


def _ols(X, y, w=None):
    """(Weighted) least squares: returns (beta, fitted, resid_sd, XtWXinv)."""
    n = len(y)
    p = len(X[0])
    ww = [1.0] * n if w is None else list(w)
    M = [[sum(ww[i] * X[i][a] * X[i][b] for i in range(n))
          for b in range(p)] for a in range(p)]
    v = [sum(ww[i] * X[i][a] * y[i] for i in range(n)) for a in range(p)]
    try:
        beta = [float(t) for t in
                np.linalg.solve(np.asarray(M, dtype=float),
                                np.asarray(v, dtype=float))]
        inv = [[float(t) for t in row] for row in
               np.linalg.inv(np.asarray(M, dtype=float))]
    except Exception:
        raise ValueError("limmav: the design matrix is singular")
    fit = [sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    df = n - p
    if df <= 0:
        raise ValueError("limmav: no residual degrees of freedom")
    rss = sum(ww[i] * (y[i] - fit[i]) ** 2 for i in range(n))
    return beta, fit, math.sqrt(rss / df), inv, df


def voom_weights(counts, design, lib_sizes=None, span=0.5):
    r"""The voom pipeline up to the weights.

    Returns ``{"log_cpm", "weights", "mean_log_count", "sqrt_sd",
    "trend_x", "trend_y", "lib_sizes"}``, where ``trend_x``/``trend_y``
    are the LOWESS knots defining :math:`lo()`.
    """
    y, R = log_cpm(counts, lib_sizes)
    G, m = len(y), len(y[0])
    X = [[float(t) for t in row] for row in design]
    if len(X) != m:
        raise ValueError("limmav: the design has %d rows but there are %d "
                         "samples" % (len(X), m))
    # step 1: OLS per gene
    fitted, sds, means = [], [], []
    for g in range(G):
        _, fit, sd, _, _ = _ols(X, y[g])
        fitted.append(fit)
        sds.append(sd)
        means.append(sum(y[g]) / m)
    # step 2: average log-cpm -> average log-count, geometric mean of R + 1
    logR = sum(math.log(v + 1.0, 2) for v in R) / m
    r_tilde = [means[g] + logR - math.log(1e6, 2) for g in range(G)]
    # step 3: LOWESS of sqrt(s) on r_tilde, read as a piecewise linear lo()
    sqrt_sd = [math.sqrt(v) for v in sds]
    smooth = lowess(r_tilde, sqrt_sd, span=span)
    order = sorted(range(G), key=lambda g: r_tilde[g])
    kx = [r_tilde[g] for g in order]
    ky = [smooth[g] for g in order]

    def lo(t):
        if t <= kx[0]:
            return ky[0]
        if t >= kx[-1]:
            return ky[-1]
        lo_i, hi_i = 0, len(kx) - 1
        while hi_i - lo_i > 1:
            mid = (lo_i + hi_i) // 2
            if kx[mid] <= t:
                lo_i = mid
            else:
                hi_i = mid
        x0, x1 = kx[lo_i], kx[hi_i]
        if x1 - x0 < 1e-15:
            return ky[lo_i]
        f = (t - x0) / (x1 - x0)
        return ky[lo_i] + f * (ky[hi_i] - ky[lo_i])

    # steps 4-5: fitted log-cpm -> fitted log-count -> weights lo()^-4
    W = []
    for g in range(G):
        row = []
        for i in range(m):
            lam = fitted[g][i] + math.log(R[i] + 1.0, 2) - math.log(1e6, 2)
            s = lo(lam)
            row.append(1.0 / (s ** 4) if s > 0 else 0.0)
        W.append(row)
    return {"log_cpm": y, "weights": W, "mean_log_count": r_tilde,
            "sqrt_sd": sqrt_sd, "trend_x": kx, "trend_y": ky,
            "lib_sizes": R, "lo": lo}


def weighted_lm(y, X, w, contrast):
    r"""Weighted least squares with a t-test on a contrast.

    Returns ``(estimate, se, t, df, sd, v_unscaled)`` where ``v_unscaled``
    is :math:`v_{gj} = c'(X'WX)^{-1}c` -- the variance factor Smyth's
    moderated statistic multiplies by :math:`\tilde s_g^2` instead of by
    the gene's own :math:`s_g^2`.
    """
    beta, fit, sd, inv, df = _ols(X, y, w)
    p = len(X[0])
    est = sum(contrast[a] * beta[a] for a in range(p))
    v_un = sum(contrast[a] * inv[a][b] * contrast[b]
               for a in range(p) for b in range(p))
    var = v_un * sd * sd
    se = math.sqrt(max(var, 0.0))
    t = est / se if se > 0 else 0.0
    return est, se, t, df, sd, v_un


def _t_sf(t, df):
    """Two-sided p-value for Student's t, by the incomplete beta."""
    x = df / (df + t * t)

    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < 1e-300:
            d = 1e-300
        d = 1.0 / d
        h = d
        for mm in range(1, 300):
            m2 = 2 * mm
            aa = mm * (b - mm) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-300:
                d = 1e-300
            c = 1.0 + aa / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            h *= d * c
            aa = -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-300:
                d = 1e-300
            c = 1.0 + aa / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < 3e-16:
                break
        return h

    a, b = 0.5 * df, 0.5
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) +
             a * math.log(x) + b * math.log(1.0 - x)) if 0 < x < 1 else None
    if lbeta is None:
        return 1.0 if x >= 1 else 0.0
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * betacf(b, a, 1.0 - x) / b


def limmav(counts, design, contrast=None, lib_sizes=None, span=0.5,
           weights=True, moderate=True):
    r"""voom-weighted differential expression.

    Parameters
    ----------
    counts : 2-D array-like
        Genes in rows, samples in columns.
    design : 2-D array-like, or sequence of group labels
        The design matrix :math:`X` (samples in rows), or labels, in which
        case an intercept plus indicators is built.
    contrast : sequence of float, optional
        Defaults to the last coefficient.
    lib_sizes : sequence of float, optional
        :math:`R_i`; column sums by default. Pass normalised library sizes
        :math:`R_i^*` here to apply the paper's optional scale
        normalisation.
    span : float
        LOWESS span for the mean-variance trend.
    weights : bool
        ``False`` runs the same pipeline unweighted, which is the
        comparison the paper's simulations make.
    moderate : bool
        Apply Smyth's (2004) empirical Bayes moderation, which is what the
        voom paper feeds these weights into. ``False`` leaves the ordinary
        weighted-least-squares t-statistics.

    Returns
    -------
    RichResult
        ``estimate`` / ``log_fold_change`` per gene (log2), with ``se``,
        ``t``, ``pvalue``, ``padj``, ``df``; ``voom_weights``, ``log_cpm``,
        ``mean_log_count``, ``sqrt_sd``, ``trend_x``, ``trend_y`` expose
        the variance model.

    Examples
    --------
    ::

        r = limmav(counts, ["A", "A", "A", "B", "B", "B"])
        r["log_fold_change"][0], r["padj"][0]

    References
    ----------
    Law, Chen, Shi & Smyth (2014) *Genome Biology* 15:R29, the voom
    variance modeling section.
    """
    first = design[0] if len(design) else None
    if isinstance(first, (list, tuple)):
        X = [[float(v) for v in row] for row in design]
    else:
        levels = []
        for lab in design:
            if lab not in levels:
                levels.append(lab)
        if len(levels) < 2:
            raise ValueError("limmav: the design has only one group")
        X = [[1.0] + [1.0 if lab == lv else 0.0 for lv in levels[1:]]
             for lab in design]
    v = voom_weights(counts, X, lib_sizes, span)
    y, W = v["log_cpm"], v["weights"]
    G, m = len(y), len(y[0])
    p = len(X[0])
    c = ([0.0] * (p - 1) + [1.0]) if contrast is None else \
        [float(t) for t in contrast]
    if len(c) != p:
        raise ValueError("limmav: the contrast must have one entry per "
                         "coefficient (%d)" % p)
    est, se, tt, pv = [], [], [], []
    sd2, vun = [], []
    df = m - p
    for g in range(G):
        e, s, t, df, sdev, v_un = weighted_lm(y[g], X,
                                              W[g] if weights else None, c)
        est.append(e)
        se.append(s)
        tt.append(t)
        sd2.append(sdev * sdev)
        vun.append(v_un)
        pv.append(_t_sf(t, df))
    eb = None
    if moderate:
        eb = ebayes(sd2, df)
        se, tt, pv = [], [], []
        for g in range(G):
            s_post = math.sqrt(eb["s2_post"][g])
            se_g = s_post * math.sqrt(max(vun[g], 0.0))
            t_g = est[g] / se_g if se_g > 0 else 0.0
            dtot = eb["df_total"][g]
            se.append(se_g)
            tt.append(t_g)
            pv.append(_t_sf(t_g, dtot) if dtot != float("inf")
                      else 2.0 * (1.0 - 0.5 * (1.0 + math.erf(
                          abs(t_g) / math.sqrt(2.0)))))
    padj = benjamini_hochberg(pv)
    return RichResult(payload={
        "estimate": est,
        "log_fold_change": est,
        "se": se,
        "t": tt,
        "pvalue": pv,
        "padj": padj,
        "df": df,
        "df_total": None if eb is None else eb["df_total"],
        "d0": None if eb is None else eb["d0"],
        "s0_sq": None if eb is None else eb["s0_sq"],
        "s2_gene": sd2,
        "s2_post": None if eb is None else eb["s2_post"],
        "moderated": bool(moderate),
        "voom_weights": W,
        "log_cpm": y,
        "mean_log_count": v["mean_log_count"],
        "sqrt_sd": v["sqrt_sd"],
        "trend_x": v["trend_x"],
        "trend_y": v["trend_y"],
        "lib_sizes": v["lib_sizes"],
        "weighted": bool(weights),
        "n_genes": G,
        "n_samples": m,
        "note": ("moderated t: gene-wise variances shrunk toward s0^2 on "
                 "d0 prior degrees of freedom and tested on d_g + d0 "
                 "(Smyth 2004)" if moderate else
                 "moderate=False: ordinary weighted-least-squares "
                 "t-statistics, no empirical Bayes"),
        "method": "voom precision weights (Law, Chen, Shi & Smyth 2014)",
    })


def cheatsheet():
    return ("limmav: voom (Law, Chen, Shi & Smyth 2014). log-cpm = "
            "log2((r + 0.5)/(R + 1) * 1e6) -- 0.5 keeps the log finite and "
            "tames low counts, 1 keeps the ratio below 1. Fit by OLS, take "
            "the residual SDs, LOWESS sqrt(s) against mean log-count "
            "(square roots because they are symmetric), read the curve as "
            "a piecewise linear lo(), map each FITTED log-cpm to a fitted "
            "log-count, and the weight is lo()^-4 -- an inverse variance, "
            "per OBSERVATION not per gene, because libraries differ in "
            "depth. Then Smyth (2004) empirical Bayes: s~^2 = (d0 s0^2 + "
            "d_g s_g^2)/(d0 + d_g), t~ = beta/(s~ sqrt(v)), tested on "
            "d_g + d0 degrees of freedom, with d0 and s0^2 from matching "
            "the first two moments of log s_g^2. d0 = 0 gives back the "
            "ordinary t; d0 = infinity gives a statistic proportional to "
            "beta.")


# compact aliases
voom = limmav
limma_voom = limmav
