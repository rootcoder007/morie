# morie.fn -- function file (rootcoder007/morie)
"""Product-of-coefficients indirect effect with the Sobel standard error."""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["ab_indirect_effect"]


def ab_indirect_effect(a, b, se_a=None, se_b=None, alpha=0.05):
    r"""Indirect effect as the product of coefficients.

    .. math:: \mathrm{IE} = ab

    where :math:`a` is the treatment-to-mediator coefficient and
    :math:`b` the mediator-to-outcome coefficient adjusted for the
    treatment. With standard errors supplied, the Sobel first-order
    delta-method standard error is

    .. math:: s_{ab} = \sqrt{b^2 s_a^2 + a^2 s_b^2}

    and :math:`ab / s_{ab}` is referred to the standard normal.

    The Sobel test is reported because it is what this quantity is
    classically paired with, but its assumption is worth stating: the
    product of two normal variables is not normal, and it is skewed
    whenever both coefficients are near zero. The test is therefore
    conservative in exactly the region where mediation is most in
    doubt, and a bootstrap or Monte Carlo interval on ``ab`` is the
    better instrument when that matters. The returned interval here is
    the symmetric normal one, so it inherits the same limitation.

    Sobel's original variance also carries a :math:`s_a^2 s_b^2` term
    that later practice usually drops. The dropped-term version is used
    here, which is the common convention and the one most software
    reports; the difference is negligible unless both standard errors
    are large relative to their coefficients.

    Parameters
    ----------
    a, b : float or array-like
        Path coefficients. Arrays are handled elementwise, which lets a
        bootstrap or posterior draw be passed straight in.
    se_a, se_b : float or array-like, optional
        Standard errors of ``a`` and ``b``. Without both, only the point
        estimate is returned and the test is skipped rather than faked.
    alpha : float, default 0.05
        Significance level for the interval.

    Returns
    -------
    RichResult
        keys: ``estimate`` (ab), ``se``, ``statistic`` (z), ``p_value``,
        ``ci_low``, ``ci_high``, ``a``, ``b``, ``method``.

    References
    ----------
    Sobel, M. E. (1982). Asymptotic confidence intervals for indirect
    effects in structural equation models. *Sociological Methodology*,
    13, 290-312.

    Baron, R. M. & Kenny, D. A. (1986). The moderator-mediator variable
    distinction in social psychological research. *Journal of
    Personality and Social Psychology*, 51(6), 1173-1182.
    """
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if not (np.all(np.isfinite(av)) and np.all(np.isfinite(bv))):
        raise ValueError("a and b must be finite.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    ie = av * bv

    if se_a is None or se_b is None:
        return RichResult(
            title="Indirect effect (product of coefficients)",
            payload={
                "estimate": ie if ie.ndim else float(ie),
                "se": None,
                "statistic": None,
                "p_value": None,
                "ci_low": None,
                "ci_high": None,
                "a": av if av.ndim else float(av),
                "b": bv if bv.ndim else float(bv),
                "method": "Indirect effect ab; no standard errors supplied, so no test",
            },
        )

    sa = np.asarray(se_a, dtype=float)
    sb = np.asarray(se_b, dtype=float)
    if not (np.all(np.isfinite(sa)) and np.all(np.isfinite(sb))):
        raise ValueError("se_a and se_b must be finite.")
    if np.any(sa < 0) or np.any(sb < 0):
        raise ValueError("Standard errors must not be negative.")

    se = np.sqrt(bv**2 * sa**2 + av**2 * sb**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, ie / se, np.nan)
    crit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = ie - crit * se, ie + crit * se
    pval = 2 * stats.norm.sf(np.abs(z))

    scalar = ie.ndim == 0
    out = (lambda v: float(v) if scalar else v)
    return RichResult(
        title="Indirect effect (Sobel)",
        payload={
            "estimate": out(ie),
            "se": out(se),
            "statistic": out(z),
            "p_value": out(pval),
            "ci_low": out(lo),
            "ci_high": out(hi),
            "a": out(av),
            "b": out(bv),
            "method": "Product-of-coefficients indirect effect, Sobel standard error",
        },
    )


def cheatsheet():
    return "abind: product-of-coefficients indirect effect (Sobel)"
