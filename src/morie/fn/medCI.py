# morie.fn -- function file (rootcoder007/morie)
"""Asymmetric confidence limits for an indirect effect."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["asymmetric_indirect_ci"]


def asymmetric_indirect_ci(a, b, sa, sb, n_sim=20000, level=0.95):
    """Confidence limits for a product, without pretending it is normal.

    The product of two normal estimates is not normal: it is skewed and
    sharply peaked, so the usual estimate-plus-or-minus-Sobel interval is
    mis-centred and too short, and the deficit is worst exactly where
    mediation studies live -- small ``a`` or small ``b``.  Taking the
    quantiles of the product distribution itself fixes the shape.

    Formula: draw ``a* ~ N(ahat, sa^2)`` and ``b* ~ N(bhat, sb^2)``
    independently, form ``a* b*``, and read off its ``alpha/2`` and
    ``1 - alpha/2`` quantiles -- MacKinnon, Lockwood & Williams (2004)
    Section 3.  The draws are a two-dimensional Halton sequence (van der
    Corput in bases 2 and 3, pushed through AS 241), so the interval is
    the same number every time and in both language arms rather than
    merely the same in distribution.

    Parameters
    ----------
    a, b : float
        Path coefficients.
    sa, sb : float
        Their standard errors, strictly positive.
    n_sim : int, default 20000
        Number of deterministic draws.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    RichResult
        ``estimate`` (the point estimate ``a b``), ``ci_lo``, ``ci_hi``,
        ``se_mc``, ``sobel_se``, ``sobel_lo``, ``sobel_hi``,
        ``asymmetry`` (upper arm minus lower arm), ``n_sim``.

    References
    ----------
    MacKinnon, D. P., Lockwood, C. M. and Williams, J. (2004).
    Confidence limits for the indirect effect: distribution of the
    product and resampling methods.  Multivariate Behavioral Research
    39(1):99-128.  doi:10.1207/s15327906mbr3901_4.
    """
    av = float(a)
    bv = float(b)
    sav = float(sa)
    sbv = float(sb)
    if sav <= 0.0 or sbv <= 0.0:
        raise ValueError("standard errors must be strictly positive")
    n = int(n_sim)
    if n < 2:
        raise ValueError("n_sim must be at least two")
    lv = float(level)
    if lv <= 0.0 or lv >= 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    za = core.normdraws(n, 2)
    zb = core.normdraws(n, 3)
    prod = sorted((av + sav * za[i]) * (bv + sbv * zb[i]) for i in range(n))
    alo = (1.0 - lv) / 2.0
    lo = core.quantile7(prod, alo)
    hi = core.quantile7(prod, 1.0 - alo)
    mean = sum(prod) / n
    var = sum((t - mean) ** 2 for t in prod) / (n - 1)
    est = av * bv
    sob = math.sqrt(av * av * sbv * sbv + bv * bv * sav * sav)
    z = core.qnorm(1.0 - alo)
    return RichResult(payload={
        "estimate": est, "ci_lo": lo, "ci_hi": hi,
        "se_mc": math.sqrt(var), "sobel_se": sob,
        "sobel_lo": est - z * sob, "sobel_hi": est + z * sob,
        "asymmetry": (hi - est) - (est - lo), "n_sim": n,
        "method": "Distribution-of-the-product confidence limits"})


def cheatsheet():
    return "medCI: asymmetric confidence limits for an indirect effect ab"
