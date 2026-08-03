# morie.fn -- shared core (rootcoder007/morie)
"""Robust estimation and hypothesis testing.

Source: Wilcox, Rand R. (2017) *Modern Statistics for the Social and
Behavioral Sciences: A Practical Introduction*, 2nd ed., CRC Press.
Equation numbers below are that book's; each was read from the PDF, and
the worked examples the book prints alongside them are the known-answer
tests in ``tests/fn/test_robust_wilcox.py``.

Everything here is plain Python -- no external numeric libraries.
"""

import math

__all__ = [
    "ideal_fourths", "idealf_iqr", "boxplot_rule",
    "trimmed_mean", "winsorize", "winsorized_mean", "winsorized_variance",
    "mad", "madn", "mad_median_rule", "mad_rescaled",
    "BOOK_MADN_CONSTANT", "R_MAD_CONSTANT",
    "yuen_test", "welch_test", "trim_counts",
    "harrell_davis", "theil_sen", "pbos",
    "percentage_bend_correlation", "winsorized_correlation",
    "mom_estimator", "one_step_m_estimator",
    "cliff_delta", "brunner_munzel", "wilcoxon_mann_whitney",
    "percentile_bootstrap_2group", "one_sample_bootstrap",
    "trimmed_mean_se", "trimmed_mean_ci", "yuen_paired",
]


# ---------------------------------------------------------------- helpers
def _flat(x):
    """Coerce to a flat list of floats, accepting any nested sequence."""
    if x is None:
        raise ValueError("expected a sequence of numbers, got None")
    out = []
    stack = [x]
    if isinstance(x, (int, float)):
        return [float(x)]
    for item in x:
        if isinstance(item, (list, tuple)):
            out.extend(_flat(item))
        else:
            out.append(float(item))
    del stack
    return out


def _median(sorted_vals):
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("median of an empty sample is undefined")
    mid = n // 2
    if n % 2:
        return float(sorted_vals[mid])
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def median(x):
    """Sample median."""
    return _median(sorted(_flat(x)))


def variance(x):
    """Sample variance s^2, the usual n-1 divisor."""
    v = _flat(x)
    n = len(v)
    if n < 2:
        raise ValueError("variance needs at least 2 observations")
    m = sum(v) / n
    return sum((t - m) ** 2 for t in v) / (n - 1)


def trim_counts(n, tr):
    """Number of observations trimmed from each tail.

    Wilcox uses g = floor(tr * n): the *integer* number of values
    removed from each end, so a 20% trimmed mean of 9 points removes
    one value per tail, not 1.8.
    """
    if not 0 <= tr < 0.5:
        raise ValueError("tr must satisfy 0 <= tr < 0.5, got %r" % (tr,))
    return int(math.floor(tr * n))


# ------------------------------------------------- ch.2 quartiles, spread
def ideal_fourths(x):
    """Lower and upper ideal fourths, eq. (2.6)-(2.7) p.27.

    With the order statistics X_(1) <= ... <= X_(n),

        j  = floor(n/4 + 5/12),      h = n/4 + 5/12 - j
        q1 = (1 - h) X_(j)   + h X_(j+1)
        q2 = (1 - h) X_(k)   + h X_(k-1),      k = n - j + 1

    These estimate the lower and upper quartiles and are what the book
    uses for outlier detection, in preference to the many alternative
    quartile definitions it lists on the same page.
    """
    v = sorted(_flat(x))
    n = len(v)
    if n < 4:
        raise ValueError("ideal fourths need at least 4 observations")
    j = int(math.floor(n / 4.0 + 5.0 / 12.0))
    h = n / 4.0 + 5.0 / 12.0 - j
    k = n - j + 1
    # the book indexes from 1; Python from 0
    q1 = (1.0 - h) * v[j - 1] + h * v[j]
    q2 = (1.0 - h) * v[k - 1] + h * v[k - 2]
    return {"q1": q1, "q2": q2, "j": j, "h": h, "k": k}


def idealf_iqr(x):
    """Interquartile range from the ideal fourths, eq. (2.8) p.27."""
    f = ideal_fourths(x)
    return f["q2"] - f["q1"]


def boxplot_rule(x, k=1.5):
    """Boxplot outlier rule, sec. 2.5.4: flag X below q1 - k*IQR or
    above q2 + k*IQR, with the fourths and IQR of eq. (2.6)-(2.8)."""
    v = _flat(x)
    f = ideal_fourths(v)
    iqr = f["q2"] - f["q1"]
    lo = f["q1"] - k * iqr
    hi = f["q2"] + k * iqr
    flags = [t < lo or t > hi for t in v]
    return {"lower": lo, "upper": hi, "iqr": iqr,
            "is_outlier": flags,
            "outliers": [t for t, b in zip(v, flags) if b],
            "n_outliers": sum(1 for b in flags if b),
            "q1": f["q1"], "q2": f["q2"]}


# --------------------------------------------- ch.2 trimming, Winsorizing
def trimmed_mean(x, tr=0.2):
    """The tr-trimmed mean, sec. 2.3.

    Removes g = floor(tr*n) values from each tail and averages the rest.
    tr = 0 gives the mean; tr -> 0.5 approaches the median.
    """
    v = sorted(_flat(x))
    n = len(v)
    g = trim_counts(n, tr)
    kept = v[g:n - g] if g else v
    if not kept:
        raise ValueError("trimming removed every observation")
    return sum(kept) / len(kept)


def winsorize(x, tr=0.2):
    """Winsorize a sample, sec. 2.2.7: the g smallest values are pulled
    up to X_(g+1) and the g largest pulled down to X_(n-g), rather than
    discarded as in trimming."""
    v = sorted(_flat(x))
    n = len(v)
    g = trim_counts(n, tr)
    if g == 0:
        return list(v)
    lo = v[g]
    hi = v[n - g - 1]
    return [min(max(t, lo), hi) for t in v]


def winsorized_mean(x, tr=0.2):
    """Winsorized mean, sec. 2.2.7: the mean of the Winsorized values."""
    w = winsorize(x, tr)
    return sum(w) / len(w)


def winsorized_variance(x, tr=0.2):
    """Winsorized variance s_w^2, sec. 2.4.5 p.28.

    "just the sample variance of the Winsorized values" -- so the n-1
    divisor of the ordinary sample variance is kept.  Its finite-sample
    breakdown point equals the amount Winsorized.
    """
    return variance(winsorize(x, tr))


# ---------------------------------------------------- ch.2 MAD and MADN
def mad(x):
    """Median absolute deviation, sec. 2.4.7 p.28: the median of
    |X_1 - M|, ..., |X_n - M| where M is the sample median."""
    v = _flat(x)
    m = median(v)
    return _median(sorted(abs(t - m) for t in v))


#: The book's MADN divisor (Wilcox sec. 2.4.7): MADN = MAD / 0.6745.
BOOK_MADN_DIVISOR = 0.6745
#: Equivalent multiplier, 1 / 0.6745.
BOOK_MADN_CONSTANT = 1.0 / 0.6745
#: R's ``mad()`` default constant, which is what WRS actually uses.
R_MAD_CONSTANT = 1.4826


def madn(x):
    """MADN = MAD / 0.6745, sec. 2.4.7.

    The divisor rescales MAD so that it estimates sigma under normality,
    which is what makes the cut-off in eq. (2.14) interpretable.

    This is the book's constant exactly.  R's ``mad()`` uses 1.4826
    instead of 1/0.6745 = 1.4825797...; see :func:`mad_rescaled` and
    :data:`R_MAD_CONSTANT` for why that matters and which estimators
    use which.
    """
    return mad(x) / BOOK_MADN_DIVISOR


def mad_median_rule(x, crit=2.24):
    """MAD-median (Hampel identifier) outlier rule, eq. (2.14) p.33.

    Declares X an outlier when |X - M| / MADN > 2.24.  Both M and MADN
    have breakdown point 0.5, so unlike the mean-and-variance rule this
    does not suffer from masking.  The book notes that 2.24 comes from
    Rousseeuw and van Zomeren (1990), and that Hampel's original used
    3.5.
    """
    v = _flat(x)
    m = median(v)
    s = madn(v)
    if s == 0:
        ratios = [0.0 if t == m else float("inf") for t in v]
    else:
        ratios = [abs(t - m) / s for t in v]
    flags = [r > crit for r in ratios]
    return {"median": m, "madn": s, "ratio": ratios,
            "is_outlier": flags,
            "outliers": [t for t, b in zip(v, flags) if b],
            "n_outliers": sum(1 for b in flags if b),
            "crit": float(crit)}


# ------------------------------------------------ ch.7 two-group methods
def _student_t_cdf(t, df):
    """CDF of Student's t, via the regularized incomplete beta."""
    x = df / (df + t * t)
    p = 0.5 * _betainc(0.5 * df, 0.5, x)
    return p if t <= 0 else 1.0 - p


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b) by continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        b * math.log1p(-x) + a * math.log(x) - lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a, b, x, itmax=300, eps=1e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def welch_test(x, y):
    """Welch's heteroscedastic test for means, sec. 7.3.

    Does not assume equal variances; the degrees of freedom are the
    Welch-Satterthwaite value.
    """
    a, b = _flat(x), _flat(y)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least 2 observations")
    m1, m2 = sum(a) / n1, sum(b) / n2
    q1 = variance(a) / n1
    q2 = variance(b) / n2
    se = math.sqrt(q1 + q2)
    tstat = (m1 - m2) / se
    df = (q1 + q2) ** 2 / (q1 * q1 / (n1 - 1) + q2 * q2 / (n2 - 1))
    p = 2.0 * (1.0 - _student_t_cdf(abs(tstat), df))
    return {"statistic": tstat, "df": df, "p_value": p, "se": se,
            "estimate": m1 - m2, "mean_x": m1, "mean_y": m2,
            "method": "Welch's heteroscedastic test for means"}


def yuen_test(x, y, tr=0.2):
    """Yuen's (1974) test for two independent trimmed means, sec. 7.4.1.

    With h the number of values left after trimming and s_w^2 the
    Winsorized variance,

        d_j = (n_j - 1) s_wj^2 / (h_j (h_j - 1))
        Ty  = (Xt_1 - Xt_2) / sqrt(d_1 + d_2)
        nu  = (d_1 + d_2)^2 / (d_1^2/(h_1-1) + d_2^2/(h_2-1))

    The book states the defining property that anchors our test: with
    no trimming this reduces exactly to Welch's method for means.
    """
    a, b = _flat(x), _flat(y)
    n1, n2 = len(a), len(b)
    g1, g2 = trim_counts(n1, tr), trim_counts(n2, tr)
    h1, h2 = n1 - 2 * g1, n2 - 2 * g2
    if h1 < 2 or h2 < 2:
        raise ValueError("too much trimming: fewer than 2 values remain")
    d1 = (n1 - 1) * winsorized_variance(a, tr) / (h1 * (h1 - 1.0))
    d2 = (n2 - 1) * winsorized_variance(b, tr) / (h2 * (h2 - 1.0))
    t1, t2 = trimmed_mean(a, tr), trimmed_mean(b, tr)
    se = math.sqrt(d1 + d2)
    tstat = (t1 - t2) / se
    df = (d1 + d2) ** 2 / (d1 * d1 / (h1 - 1.0) + d2 * d2 / (h2 - 1.0))
    p = 2.0 * (1.0 - _student_t_cdf(abs(tstat), df))
    return {"statistic": tstat, "df": df, "p_value": p, "se": se,
            "estimate": t1 - t2, "trimmed_mean_x": t1,
            "trimmed_mean_y": t2, "h_x": h1, "h_y": h2, "tr": float(tr),
            "method": "Yuen's test for two independent trimmed means"}


# ============================================================
# Estimators verified against Wilcox's own R implementation,
# WRS Rallfun-v45.R (github.com/nicebread/WRS), the reference
# code for the book.  A copy of the exact functions we ported
# is kept at ledger/shelves/WRS_REFERENCE_FUNCTIONS.R.
# ============================================================

def _pbeta(q, a, b):
    """Beta CDF, i.e. R's pbeta(q, a, b)."""
    return _betainc(a, b, q)


def mad_rescaled(x, constant=R_MAD_CONSTANT):
    """MAD rescaled the way R's ``mad()`` does it.

    Wilcox's WRS code calls R's ``mad()``, whose default constant is
    1.4826, and his own comment there reads "mad in splus is madn in
    the book".  So R's ``mad()`` is the book's MADN.  The two differ in
    the last few digits only -- 1/0.6745 = 1.4825797 -- but the
    estimators below are ported from the R code, so they use the R
    constant and :func:`madn` keeps the book's.
    """
    v = _flat(x)
    m = median(v)
    return _median(sorted(abs(t - m) for t in v)) * float(constant)


def harrell_davis(x, q=0.5):
    """Harrell-Davis quantile estimator.

    A weighted average of all the order statistics, the weights being
    increments of a Beta((n+1)q, (n+1)(1-q)) distribution function:

        w_i = I_{i/n}(m1, m2) - I_{(i-1)/n}(m1, m2)
        theta_q = sum_i w_i X_(i)

    Because every observation contributes, it is usually more efficient
    than a single order statistic, which matters most in the tails.
    Ported from WRS ``hd``.
    """
    v = sorted(_flat(x))
    n = len(v)
    if n == 0:
        raise ValueError("need at least one observation")
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between 0 and 1")
    m1 = (n + 1) * q
    m2 = (n + 1) * (1.0 - q)
    total = 0.0
    prev = _pbeta(0.0, m1, m2)
    for i in range(1, n + 1):
        cur = _pbeta(i / n, m1, m2)
        total += (cur - prev) * v[i - 1]
        prev = cur
    return total


def theil_sen(x, y, intercept_at_medians=False):
    """Theil-Sen regression.

    The slope is the median of the pairwise slopes

        (Y_j - Y_i) / (X_j - X_i)   over all pairs with X_j > X_i,

    which has a 29% breakdown point against the least-squares 0%.
    ``intercept_at_medians`` selects between the two intercepts WRS
    offers: False (the default, matching WRS since Rallfun-v29 and the
    other R implementations) uses median(Y - slope X); True uses
    median(Y) - slope median(X).  Ported from WRS ``tsp1reg``.
    """
    xs = _flat(x)
    ys = _flat(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 observations")
    slopes = []
    for i in range(n):
        for j in range(n):
            dx = xs[j] - xs[i]
            if dx > 0:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        raise ValueError("no two observations have distinct x values")
    slope = _median(sorted(slopes))
    if intercept_at_medians:
        inter = median(ys) - slope * median(xs)
    else:
        inter = _median(sorted(ys[i] - slope * xs[i] for i in range(n)))
    resid = [ys[i] - slope * xs[i] - inter for i in range(n)]
    return {"slope": slope, "intercept": inter, "coef": [inter, slope],
            "residuals": resid, "n_pairs": len(slopes),
            "method": "Theil-Sen regression"}


def pbos(x, beta=0.2):
    """One-step percentage bend measure of location.

    With omega the floor((1-beta) n)-th smallest |X - M|, values whose
    standardised deviation exceeds 1 in absolute value are pulled to
    the boundary and the estimate corrects for how many were pulled
    from each side.  Ported from WRS ``pbos``.
    """
    v = _flat(x)
    n = len(v)
    m = median(v)
    dev = sorted(abs(t - m) for t in v)
    idx = int(math.floor((1.0 - beta) * n))
    omega = dev[idx - 1] if idx >= 1 else dev[0]
    if omega == 0:
        return m
    psi = [(t - m) / omega for t in v]
    i1 = sum(1 for p in psi if p < -1)
    i2 = sum(1 for p in psi if p > 1)
    kept = sum(v[k] for k in range(n) if -1 <= psi[k] <= 1)
    denom = n - i1 - i2
    if denom == 0:
        return m
    return (kept + omega * (i2 - i1)) / denom


def percentage_bend_correlation(x, y, beta=0.2):
    """Percentage bend correlation.

    Each variable is centred at its percentage bend location, scaled by
    its omega, and clipped to [-1, 1]; the correlation of the clipped
    values is the estimate.  Ported from WRS ``pbcor``.
    """
    xs, ys = _flat(x), _flat(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = len(xs)

    def _scaled(v):
        m = median(v)
        dev = sorted(abs(t - m) for t in v)
        idx = int(math.floor((1.0 - beta) * len(v)))
        omega = dev[idx - 1] if idx >= 1 else dev[0]
        if omega == 0:
            raise ValueError("omega is zero; the data are too tied")
        loc = pbos(v, beta)
        out = []
        for t in v:
            s = (t - loc) / omega
            out.append(-1.0 if s <= -1 else (1.0 if s >= 1 else s))
        return out

    a = _scaled(xs)
    b = _scaled(ys)
    den = math.sqrt(sum(t * t for t in a) * sum(t * t for t in b))
    if den == 0:
        raise ValueError("degenerate data: zero scale")
    r = sum(a[i] * b[i] for i in range(n)) / den
    if abs(r) >= 1.0:
        stat, p = float("inf") * (1 if r > 0 else -1), 0.0
    else:
        stat = r * math.sqrt((n - 2) / (1 - r * r))
        p = 2.0 * (1.0 - _student_t_cdf(abs(stat), n - 2))
    return {"cor": r, "statistic": stat, "p_value": p, "n": n,
            "method": "percentage bend correlation"}


def winsorized_correlation(x, y, tr=0.2):
    """Winsorized correlation.

    The Pearson correlation of the Winsorized values.  The test
    statistic uses n - 2g - 2 degrees of freedom, g being the number
    Winsorized in each tail, because Winsorizing removes that much
    independent information.  Ported from WRS ``wincor``.
    """
    xs, ys = _flat(x), _flat(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = len(xs)
    g = trim_counts(n, tr)
    a = _winsorize_paired(xs, tr)
    b = _winsorize_paired(ys, tr)
    ma = sum(a) / n
    mb = sum(b) / n
    sa = math.sqrt(sum((t - ma) ** 2 for t in a))
    sb = math.sqrt(sum((t - mb) ** 2 for t in b))
    if sa == 0 or sb == 0:
        raise ValueError("degenerate data: zero Winsorized variance")
    r = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (sa * sb)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)
    df = n - 2 * g - 2
    if abs(r) >= 1.0 or df <= 0:
        stat, p = float("nan"), float("nan")
    else:
        stat = r * math.sqrt((n - 2) / (1 - r * r))
        p = 2.0 * (1.0 - _student_t_cdf(abs(stat), df))
    return {"cor": r, "cov": cov, "statistic": stat, "p_value": p,
            "n": n, "df": df, "method": "Winsorized correlation"}


def _winsorize_paired(v, tr):
    """Winsorize keeping the ORIGINAL order, which the correlations
    need -- :func:`winsorize` returns the sorted sample."""
    y = sorted(v)
    n = len(v)
    ibot = int(math.floor(tr * n))          # 0-based index of y[g+1] in R
    itop = n - ibot - 1
    lo, hi = y[ibot], y[itop]
    return [min(max(t, lo), hi) for t in v]


def mom_estimator(x, bend=2.24, constant=R_MAD_CONSTANT):
    """Modified one-step M-estimator (MOM) of location.

    Drops every value more than ``bend`` MADNs from the median and
    averages what is left; the 2.24 cut-off is the same one the
    MAD-median outlier rule uses.  Ported from WRS ``mom``.

    ``constant`` scales the MAD.  It defaults to R's 1.4826, which is
    what WRS uses, so this reproduces ``mom()`` exactly.  Pass
    :data:`BOOK_MADN_CONSTANT` to follow the book's MAD/0.6745 instead;
    the two differ in the fifth significant figure and can flip which
    observations fall on the boundary.
    """
    v = _flat(x)
    m = median(v)
    s = mad_rescaled(v, constant)
    if s == 0:
        return m
    kept = [t for t in v if m - bend * s <= t <= m + bend * s]
    if not kept:
        return m
    return sum(kept) / len(kept)


def _huber_psi(u, bend=1.28):
    return u if abs(u) <= bend else bend * (1.0 if u > 0 else -1.0)


def one_step_m_estimator(x, bend=1.28, constant=R_MAD_CONSTANT):
    """One-step M-estimator of location with Huber's Psi.

        theta = M + MADN * sum(psi(y_i)) / #{|y_i| <= bend},
        y_i = (X_i - M) / MADN

    Ported from WRS ``onestep``; the default bend 1.28 is Wilcox's.

    ``constant`` scales the MAD, defaulting to R's 1.4826 as WRS does.
    Pass :data:`BOOK_MADN_CONSTANT` for the book's MAD/0.6745.
    """
    v = _flat(x)
    m = median(v)
    s = mad_rescaled(v, constant)
    if s == 0:
        return m
    y = [(t - m) / s for t in v]
    a = sum(_huber_psi(t, bend) for t in y)
    b = sum(1 for t in y if abs(t) <= bend)
    if b == 0:
        return m
    return m + s * a / b


def cliff_delta(x, y, alpha=0.05):
    """Cliff's delta and a confidence interval for P(X<Y).

    delta = P(X>Y) - P(X<Y), estimated by the mean of sign(X_i - Y_j)
    over all pairs.  The interval is Cliff (1996, p.140, eq. 5.12),
    which is asymmetric in delta and handles ties.  Ported from WRS
    ``cid``.
    """
    xs, ys = _flat(x), _flat(y)
    n1, n2 = len(xs), len(ys)
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least 2 observations")
    signs = [[(1 if a > b else (-1 if a < b else 0)) for b in ys]
             for a in xs]
    flat = [s for row in signs for s in row]
    d = sum(flat) / (n1 * n2)
    phat = (1.0 - d) / 2.0
    p_less = sum(1 for s in flat if s < 0) / (n1 * n2)
    p_eq = sum(1 for s in flat if s == 0) / (n1 * n2)
    p_gt = sum(1 for s in flat if s > 0) / (n1 * n2)
    out = {"delta": d, "p_hat": phat, "n1": n1, "n2": n2,
           "P_x_less_y": p_less, "P_equal": p_eq, "P_x_greater_y": p_gt,
           "method": "Cliff's delta"}
    if phat in (0.0, 1.0):
        out["ci"] = (float("nan"), float("nan"))
        return out
    sigdih = sum((s - d) ** 2 for s in flat) / (n1 * n2 - 1)
    di = [sum(1 for b in ys if a > b) / n2 - sum(1 for b in ys if a < b) / n2
          for a in xs]
    dh = [sum(1 for a in xs if b > a) / n1 - sum(1 for a in xs if b < a) / n1
          for b in ys]
    sdi = variance(di)
    sdh = variance(dh)
    sh = ((n2 - 1) * sdi + (n1 - 1) * sdh + sigdih) / (n1 * n2)
    zv = -_norm_quantile(1.0 - alpha / 2.0)
    root = math.sqrt(sh) * math.sqrt((1 - d * d) ** 2 + zv * zv * sh)
    den = 1 - d * d + zv * zv * sh
    cu = (d - d ** 3 - zv * root) / den
    cl = (d - d ** 3 + zv * root) / den
    out["ci"] = (cl, cu)
    return out


def _norm_quantile(p):
    """Standard normal quantile, Acklam's rational approximation
    refined by one Halley step against the erf-based CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= 1 - pl:
        q = p - 0.5
        r = q * q
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
             + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3])
                             * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
              + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    e = 0.5 * math.erfc(-z / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(z * z / 2)
    return z - u / (1 + z * u / 2)


def brunner_munzel(x, y, alpha=0.05):
    """Brunner-Munzel (2000) heteroscedastic rank test.

    The heteroscedastic analogue of Wilcoxon-Mann-Whitney: it tests
    P(X<Y) + 0.5 P(X=Y) = 1/2 without assuming the two distributions
    have the same shape, which is what WMW needs and rarely has.
    Ported from WRS ``bmp``.
    """
    xs, ys = _flat(x), _flat(y)
    n1, n2 = len(xs), len(ys)
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least 2 observations")
    N = n1 + n2
    R = _rank(xs + ys)
    R1 = sum(R[:n1]) / n1
    R2 = sum(R[n1:]) / n2
    Rg1 = _rank(xs)
    Rg2 = _rank(ys)
    s1 = sum((R[i] - Rg1[i] - R1 + (n1 + 1) / 2.0) ** 2
             for i in range(n1)) / (n1 - 1)
    s2 = sum((R[n1 + j] - Rg2[j] - R2 + (n2 + 1) / 2.0) ** 2
             for j in range(n2)) / (n2 - 1)
    se = math.sqrt(N) * math.sqrt(N * (s1 / (n2 * n2) / n1
                                       + s2 / (n1 * n1) / n2))
    phat = (R2 - (n2 + 1) / 2.0) / n1
    if se == 0:
        # Complete separation: every value of one group beats every
        # value of the other, so both within-group rank variances
        # vanish and the statistic is undefined.  WRS `bmp` detects the
        # same case (phat == 0 or 1) and falls back to a binomial
        # interval rather than a t interval.  We report the separation
        # instead of inventing a finite statistic.
        return {"statistic": float("inf") if phat > 0.5
                else float("-inf"),
                "df": float("nan"), "p_value": float("nan"),
                "p_hat": phat, "delta": 1.0 - 2.0 * phat, "se": 0.0,
                "n1": n1, "n2": n2, "separated": True,
                "method": "Brunner-Munzel test (complete separation)"}
    stat = (R2 - R1) / se
    den = (s1 / n2) ** 2 / (n1 - 1) + (s2 / n1) ** 2 / (n2 - 1)
    df = (s1 / n2 + s2 / n1) ** 2 / den if den > 0 else float("nan")
    p = 2.0 * (1.0 - _student_t_cdf(abs(stat), df))
    return {"statistic": stat, "df": df, "p_value": p, "p_hat": phat,
            "delta": 1.0 - 2.0 * phat, "se": se, "n1": n1, "n2": n2,
            "separated": False, "method": "Brunner-Munzel test"}


def _rank(v):
    """Ranks with ties averaged, i.e. R's ``rank()`` default."""
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def wilcoxon_mann_whitney(x, y):
    """Wilcoxon-Mann-Whitney rank-sum test, normal approximation with
    a tie correction.

    Wilcox's point about this test is worth repeating: it tests
    P(X<Y) = 1/2 only under the assumption of identical distributions;
    when the groups are heteroscedastic use :func:`brunner_munzel`
    instead.
    """
    xs, ys = _flat(x), _flat(y)
    n1, n2 = len(xs), len(ys)
    if n1 < 1 or n2 < 1:
        raise ValueError("both groups must be non-empty")
    R = _rank(xs + ys)
    W = sum(R[:n1])
    mu = n1 * (n1 + n2 + 1) / 2.0
    counts = {}
    for t in xs + ys:
        counts[t] = counts.get(t, 0) + 1
    tie = sum(c ** 3 - c for c in counts.values())
    N = n1 + n2
    var = n1 * n2 / 12.0 * ((N + 1) - tie / float(N * (N - 1)))
    if var <= 0:
        raise ValueError("zero variance: all values are tied")
    z = (W - mu) / math.sqrt(var)
    p = 2.0 * (1.0 - 0.5 * math.erfc(-abs(z) / math.sqrt(2)))
    U = W - n1 * (n1 + 1) / 2.0
    return {"statistic": W, "U": U, "z": z, "p_value": p,
            "n1": n1, "n2": n2,
            "method": "Wilcoxon-Mann-Whitney rank-sum test"}


def percentile_bootstrap_2group(x, y, est=None, nboot=2000, alpha=0.05,
                                seed=2, **kwargs):
    """Percentile bootstrap for the difference between two estimators.

    Resamples each group with replacement, forms the distribution of
    the difference in the chosen estimator, and reads the interval off
    its percentiles; the p-value is twice the smaller tail proportion.
    ``est`` defaults to the one-step M-estimator, as in WRS ``pb2gen``.
    """
    import random

    xs, ys = _flat(x), _flat(y)
    if est is None:
        est = one_step_m_estimator
    rng = random.Random(seed)
    n1, n2 = len(xs), len(ys)
    diffs = []
    for _ in range(int(nboot)):
        bx = [xs[rng.randrange(n1)] for _ in range(n1)]
        by = [ys[rng.randrange(n2)] for _ in range(n2)]
        diffs.append(est(bx, **kwargs) - est(by, **kwargs))
    diffs.sort()
    low = int(round((alpha / 2.0) * nboot))
    up = int(nboot - low) - 1
    low = min(max(low, 0), nboot - 1)
    up = min(max(up, 0), nboot - 1)
    temp = (sum(1 for v in diffs if v < 0) / nboot
            + sum(1 for v in diffs if v == 0) / (2.0 * nboot))
    p = 2.0 * min(temp, 1.0 - temp)
    e1, e2 = est(xs, **kwargs), est(ys, **kwargs)
    return {"est_1": e1, "est_2": e2, "est_diff": e1 - e2,
            "ci": (diffs[low], diffs[up]), "p_value": p,
            "n1": n1, "n2": n2, "nboot": int(nboot),
            "method": "percentile bootstrap for a difference"}


def _student_t_quantile(p, df, tol=1e-12, max_iter=200):
    """Inverse of :func:`_student_t_cdf`, i.e. R's ``qt``.

    Bisection on the CDF: the CDF is smooth and strictly increasing, so
    this converges reliably without needing a series expansion.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    if p == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while _student_t_cdf(lo, df) > p:
        lo *= 2.0
        if lo < -1e12:
            break
    while _student_t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def trimmed_mean_se(x, tr=0.2):
    """Standard error of the trimmed mean (Tukey-McLaughlin).

        SE = sqrt(s_w^2) / ((1 - 2 tr) sqrt(n))

    The Winsorized variance appears because the trimmed mean's sampling
    variance depends on the Winsorized, not the ordinary, spread.
    Ported from WRS ``trimse``.
    """
    v = _flat(x)
    return math.sqrt(winsorized_variance(v, tr)) / (
        (1.0 - 2.0 * tr) * math.sqrt(len(v)))


def trimmed_mean_ci(x, tr=0.2, alpha=0.05, null_value=0.0):
    """Confidence interval and test for a single trimmed mean.

    Uses the Tukey-McLaughlin standard error with
    df = n - 2 floor(tr n) - 1.  Ported from WRS ``trimci``.
    """
    v = _flat(x)
    n = len(v)
    se = trimmed_mean_se(v, tr)
    df = n - 2 * trim_counts(n, tr) - 1
    if df <= 0:
        raise ValueError("too much trimming: no degrees of freedom left")
    est = trimmed_mean(v, tr)
    crit = _student_t_quantile(1.0 - alpha / 2.0, df)
    stat = (est - float(null_value)) / se
    p = 2.0 * (1.0 - _student_t_cdf(abs(stat), df))
    return {"estimate": est, "ci": (est - crit * se, est + crit * se),
            "statistic": stat, "se": se, "df": df, "p_value": p, "n": n,
            "method": "trimmed-mean confidence interval"}


def yuen_paired(x, y, tr=0.2, alpha=0.05):
    """Yuen's test for two DEPENDENT trimmed means.

    Unlike the independent-groups version the standard error must
    subtract the Winsorized covariance, since the pairs are related:

        se = sqrt((q1 + q2 - 2 q3) / (h (h - 1)))

    with q1, q2 the (n-1)-scaled Winsorized variances, q3 the
    (n-1)-scaled Winsorized covariance and h the number of untrimmed
    observations.  Ported from WRS ``yuend``.
    """
    xs, ys = _flat(x), _flat(y)
    if len(xs) != len(ys):
        raise ValueError("dependent groups must have equal length")
    n = len(xs)
    h1 = n - 2 * trim_counts(n, tr)
    if h1 < 2:
        raise ValueError("too much trimming: fewer than 2 values remain")
    q1 = (n - 1) * winsorized_variance(xs, tr)
    q2 = (n - 1) * winsorized_variance(ys, tr)
    q3 = (n - 1) * winsorized_correlation(xs, ys, tr)["cov"]
    df = h1 - 1
    var = (q1 + q2 - 2.0 * q3) / (h1 * (h1 - 1.0))
    if var <= 0:
        # q1 + q2 - 2 q3 vanishes when the two Winsorized samples move
        # in lockstep, e.g. y = x + c.  The paired differences then have
        # no Winsorized variability at all, so the statistic is
        # undefined rather than infinite -- report that instead of
        # dividing by zero.
        dif0 = trimmed_mean(xs, tr) - trimmed_mean(ys, tr)
        return {"estimate": dif0, "ci": (dif0, dif0),
                "statistic": float("nan"), "se": 0.0, "df": df,
                "p_value": float("nan"), "n": n,
                "est_1": trimmed_mean(xs, tr),
                "est_2": trimmed_mean(ys, tr), "degenerate": True,
                "method": "Yuen's test for dependent trimmed means "
                          "(zero Winsorized variance of the differences)"}
    se = math.sqrt(var)
    dif = trimmed_mean(xs, tr) - trimmed_mean(ys, tr)
    crit = _student_t_quantile(1.0 - alpha / 2.0, df)
    stat = dif / se
    p = 2.0 * (1.0 - _student_t_cdf(abs(stat), df))
    return {"estimate": dif, "ci": (dif - crit * se, dif + crit * se),
            "statistic": stat, "se": se, "df": df, "p_value": p, "n": n,
            "est_1": trimmed_mean(xs, tr), "est_2": trimmed_mean(ys, tr),
            "degenerate": False,
            "method": "Yuen's test for dependent trimmed means"}


def one_sample_bootstrap(x, est=None, alpha=0.05, nboot=2000,
                         null_value=0.0, seed=2, **kwargs):
    """Percentile bootstrap interval for a single measure of location.

    ``est`` defaults to the one-step M-estimator, as in WRS
    ``onesampb``.  The p-value is twice the smaller of the proportions
    of bootstrap estimates above and below the null value.
    """
    import random

    v = _flat(x)
    if est is None:
        est = one_step_m_estimator
    rng = random.Random(seed)
    n = len(v)
    boot = sorted(est([v[rng.randrange(n)] for _ in range(n)], **kwargs)
                  for _ in range(int(nboot)))
    low = int(round((alpha / 2.0) * nboot))
    up = int(nboot - low) - 1
    low = min(max(low, 0), nboot - 1)
    up = min(max(up, 0), nboot - 1)
    nv = float(null_value)
    above = sum(1 for b in boot if b > nv) / nboot
    equal = sum(1 for b in boot if b == nv) / nboot
    pv = above + 0.5 * equal
    p = 2.0 * min(pv, 1.0 - pv)
    return {"estimate": est(v, **kwargs), "ci": (boot[low], boot[up]),
            "p_value": p, "n": n, "nboot": int(nboot),
            "method": "one-sample percentile bootstrap"}
