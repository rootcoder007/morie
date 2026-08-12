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

What follows the weighting here is weighted least squares with ordinary
t-statistics and Benjamini-Hochberg adjustment. limma's **empirical Bayes
moderation** of the gene-wise variances (Smyth 2004) is what the paper feeds
these weights into and is *not* implemented -- that is a different paper, not
in this library, and moderated t-statistics are not the same as the ordinary
ones. The result says so rather than letting the caller assume otherwise.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

from .deseq2 import benjamini_hochberg

__all__ = ["limmav", "voom", "limma_voom", "log_cpm", "lowess", "voom_weights",
           "weighted_lm"]


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
    """Weighted least squares with an ordinary t-test on a contrast."""
    beta, fit, sd, inv, df = _ols(X, y, w)
    p = len(X[0])
    est = sum(contrast[a] * beta[a] for a in range(p))
    var = sum(contrast[a] * inv[a][b] * contrast[b]
              for a in range(p) for b in range(p)) * sd * sd
    se = math.sqrt(max(var, 0.0))
    t = est / se if se > 0 else 0.0
    return est, se, t, df, sd


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
           weights=True):
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
    df = m - p
    for g in range(G):
        e, s, t, df, _ = weighted_lm(y[g], X, W[g] if weights else None, c)
        est.append(e)
        se.append(s)
        tt.append(t)
        pv.append(_t_sf(t, df))
    padj = benjamini_hochberg(pv)
    return RichResult(payload={
        "estimate": est,
        "log_fold_change": est,
        "se": se,
        "t": tt,
        "pvalue": pv,
        "padj": padj,
        "df": df,
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
        "note": "limma's empirical Bayes variance moderation (Smyth 2004) "
                "is NOT applied; these are ordinary weighted-least-squares "
                "t-statistics",
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
            "depth. Empirical Bayes moderation is NOT here.")


# compact aliases
voom = limmav
limma_voom = limmav
