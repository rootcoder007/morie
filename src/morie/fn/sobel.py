"""Sobel test for an indirect (mediated) effect."""

from math import sqrt

from . import _stats_core as stats
from ._richresult import hypothesis_test_result

__all__ = ["sobel_test"]

_VARIANTS = ("sobel", "aroian", "goodman")


def sobel_test(a, b, se_a, se_b, variant="sobel"):
    r"""Test the indirect effect ``a*b`` in a mediation model.

    With ``a`` the X -> M path and ``b`` the M -> Y path, the indirect
    effect is their product. The delta method gives its variance from
    the first-order Taylor expansion of ``f(a,b) = ab``, whose gradient
    is ``(b, a)``:

    .. math::

       \operatorname{Var}(ab)\approx b^{2}\sigma_a^{2}+a^{2}\sigma_b^{2},
       \qquad
       z=\frac{ab}{\sqrt{b^{2}\sigma_a^{2}+a^{2}\sigma_b^{2}}} .

    That is Sobel's form. The exact variance of a product of two
    independent normals carries a third term, and the two classical
    variants differ only in its sign:

    ================  ==========================================
    ``variant``       variance
    ================  ==========================================
    ``"sobel"``       :math:`b^2\sigma_a^2+a^2\sigma_b^2`
    ``"aroian"``      :math:`b^2\sigma_a^2+a^2\sigma_b^2+\sigma_a^2\sigma_b^2`
    ``"goodman"``     :math:`b^2\sigma_a^2+a^2\sigma_b^2-\sigma_a^2\sigma_b^2`
    ================  ==========================================

    Aroian's is the exact variance under independence; Goodman's is the
    unbiased estimator and can go negative, in which case no z exists
    and this raises rather than returning a fabricated number.

    The p-value assumes ``ab`` is normal. It is not -- a product of
    normals is heavy-tailed and skewed -- so this test is
    under-powered and its confidence interval is symmetric when the
    true one is not. That is a property of the method, not of this
    implementation; the distribution-of-products or bootstrap
    approaches exist precisely because of it.

    Parameters
    ----------
    a, b : float
        Path coefficients.
    se_a, se_b : float
        Their standard errors.
    variant : {"sobel", "aroian", "goodman"}

    Returns
    -------
    RichResult
        Keys ``statistic`` (z), ``pvalue`` (two-sided), ``indirect_effect``,
        ``se``, ``variant``, ``ci_lower``, ``ci_upper`` (95%, normal).

    References
    ----------
    Sobel, M. E. (1982). Asymptotic confidence intervals for indirect
    effects in structural equation models. *Sociological Methodology*,
    13, 290-312.
    Aroian, L. A. (1947). The probability function of the product of
    two normally distributed variables. *Annals of Mathematical
    Statistics*, 18(2), 265-271.
    Goodman, L. A. (1960). On the exact variance of products.
    *Journal of the American Statistical Association*, 55, 708-713.
    """
    if variant not in _VARIANTS:
        raise ValueError("variant must be one of %s" % (_VARIANTS,))
    a = float(a)
    b = float(b)
    va = float(se_a) ** 2
    vb = float(se_b) ** 2
    if se_a < 0 or se_b < 0:
        raise ValueError("standard errors must be non-negative.")
    var = b * b * va + a * a * vb
    if variant == "aroian":
        var += va * vb
    elif variant == "goodman":
        var -= va * vb
    if var <= 0:
        raise ValueError(
            "non-positive variance for the indirect effect (variant=%r); "
            "no z statistic exists." % variant
        )
    se = sqrt(var)
    est = a * b
    z = est / se
    return hypothesis_test_result(
        test_name="Sobel test for an indirect effect",
        statistic=float(z),
        pvalue=float(2.0 * stats.norm.sf(abs(z))),
        extra_summary=[("indirect_effect", est), ("se", se)],
        extra_payload={
            "indirect_effect": float(est),
            "se": float(se),
            "variant": variant,
            "a": a, "b": b, "se_a": float(se_a), "se_b": float(se_b),
            "ci_lower": float(est - 1.959963984540054 * se),
            "ci_upper": float(est + 1.959963984540054 * se),
            "method": "Sobel (1982) delta-method test of a*b (%s variance)" % variant,
        },
    )


def cheatsheet():
    return "sobel: Sobel/Aroian/Goodman test of the indirect effect a*b"
