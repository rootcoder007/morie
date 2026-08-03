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
    "mad", "madn", "mad_median_rule",
    "yuen_test", "welch_test", "trim_counts",
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


def madn(x):
    """MADN = MAD / 0.6745, sec. 2.4.7.

    The divisor rescales MAD so that it estimates sigma under normality,
    which is what makes the cut-off in eq. (2.14) interpretable.
    """
    return mad(x) / 0.6745


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
