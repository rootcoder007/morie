# morie.fn -- function file (rootcoder007/morie)
"""
Lilliefors test for normality.

Tests whether a sample comes from a normal distribution when parameters
are estimated from the data (less conservative than K-S one-sample).

Reference: Gibbons & Chakraborti (2011), Nonparametric Statistical Inference, 5th Ed. § 4.3
"""

import math

__all__ = ["lilef"]


def _dallal_wilkinson_p(d, n):
    """Lilliefors p-value: Dallal-Wilkinson (1986) analytic approximation for
    p < 0.1, Stephens' (1974) modified statistic above it (as nortest)."""
    if n <= 100:
        kd, nd = d, float(n)
    else:
        kd, nd = d * (n / 100.0) ** 0.49, 100.0
    p = math.exp(
        -7.01256 * kd * kd * (nd + 2.78019)
        + 2.99587 * kd * math.sqrt(nd + 2.78019)
        - 0.122119
        + 0.974598 / math.sqrt(nd)
        + 1.67997 / nd
    )
    if p > 0.1:
        kk = (math.sqrt(n) - 0.01 + 0.85 / math.sqrt(n)) * d
        if kk <= 0.302:
            p = 1.0
        elif kk <= 0.5:
            p = 2.76773 - 19.828315 * kk + 80.709644 * kk**2 - 138.55152 * kk**3 + 81.218052 * kk**4
        elif kk <= 0.9:
            p = -4.901232 + 40.662806 * kk - 97.490286 * kk**2 + 94.029866 * kk**3 - 32.355711 * kk**4
        elif kk <= 1.31:
            p = 6.198765 - 19.558097 * kk + 23.186922 * kk**2 - 12.234627 * kk**3 + 2.423045 * kk**4
        else:
            p = 0.0
    return p


def lilef(x, axis=0, cdf=None):
    r"""
    Lilliefors test for normality.

    The Kolmogorov-Smirnov distance to the normal cdf with the sample mean
    and standard deviation plugged in,
    :math:`D = \max(D^+, D^-)`, :math:`D^+ = \max_i (i/n - \Phi(z_{(i)}))`,
    :math:`D^- = \max_i (\Phi(z_{(i)}) - (i-1)/n)`. Estimating the two
    parameters makes the null distribution of :math:`D` differ from the
    Kolmogorov one; its p-value is the Dallal and Wilkinson (1986) analytic
    approximation (valid below 0.1), switching to Stephens' (1974) modified
    statistic :math:`(\sqrt n - 0.01 + 0.85/\sqrt n) D` above it -- the
    rule used by ``nortest::lillie.test``.

    Parameters
    ----------
    x : array_like
        Input sample, n >= 4.
    axis : int, optional
        For a 2-D input, the first slice along this axis is tested.
    cdf : ignored
        Kept for signature stability; the null is always the normal.

    Returns
    -------
    dict
        ``statistic`` (D), ``p_value``, ``critical_value`` (Lilliefors'
        large-sample 5 percent point 0.886/sqrt(n)), ``interpretation``
        ("reject" when p < 0.05), ``mean``, ``std`` (divisor n - 1).

    References
    ----------
    Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality
    with mean and variance unknown. JASA 62, 399-402.
    Dallal, G. E. and Wilkinson, L. (1986). An analytic approximation to the
    distribution of Lilliefors's test statistic for normality. The American
    Statistician 40, 294-296.
    Stephens, M. A. (1974). EDF statistics for goodness of fit and some
    comparisons. JASA 69, 730-737.

    Examples
    --------
    >>> r = lilef([2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 9.9, 2.5, 3.3, 2.7])
    >>> round(r["statistic"], 10), round(r["p_value"], 10)
    (0.3547551991, 0.0008043292)
    """
    if hasattr(x, "tolist"):
        x = x.tolist()
    x = list(x)
    if x and isinstance(x[0], (list, tuple)):
        x = list(x[0]) if axis == 0 else [row[0] for row in x]
    v = [float(t) for t in x]
    n = len(v)
    if n < 4:
        raise ValueError("Sample size must be at least 4 for Lilliefors test")
    mean = math.fsum(v) / n
    std = math.sqrt(math.fsum((t - mean) ** 2 for t in v) / (n - 1))
    if std <= 0:
        raise ValueError("Sample has zero variance; the normal fit is degenerate")
    zs = sorted((t - mean) / std for t in v)
    F = [0.5 * math.erfc(-z / math.sqrt(2.0)) for z in zs]
    D = max(max((i + 1) / n - F[i], F[i] - i / n) for i in range(n))
    p_value = _dallal_wilkinson_p(D, n)
    critical_value = 0.886 / math.sqrt(n)
    return {
        "statistic": float(D),
        "p_value": float(p_value),
        "critical_value": float(critical_value),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
        "mean": float(mean),
        "std": float(std),
    }


def cheatsheet() -> str:
    return "lilef: lilef(x, axis, cdf) -> Lilliefors test for normality (Dallal-Wilkinson p-value)."
