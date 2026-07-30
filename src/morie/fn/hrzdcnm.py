# morie.fn -- function file (rootcoder007/morie)
"""Deconvolution asymptotic normality."""

import numpy as np

from ._richresult import RichResult

__all__ = ["hrz_deconv_normality", "horowitz_deconv_normality"]


def hrz_deconv_normality(fn_u, f_u, n, h, b, bias=0.0, sigma=1.0):
    r"""Asymptotic normality of the deconvolution estimator (Horowitz
    Sec. 5.1.3):

    .. math:: \left[\frac{n h_n}{b_n}\right]^{1/2}
              \big(\hat f_U(u) - f_U(u) - \mathrm{bias}\big)
              \;\to_D\; N(0, \sigma^2).

    The normalising factor carries :math:`b_n`, a deconvolution-specific
    inflation absent from ordinary kernel estimation -- it is what
    encodes the price of dividing by a vanishing characteristic
    function. The bias term is SUBTRACTED, not assumed away: an
    undersmoothed bandwidth is what makes it negligible, and if it is
    not, the interval is centred wrongly.

    Parameters
    ----------
    fn_u, f_u : float
        Estimate and truth at the evaluation point.
    n : int
        Sample size.
    h, b : float > 0
        Bandwidth and the deconvolution inflation factor.
    bias : float, default 0.0
        Asymptotic bias.
    sigma : float, default 1.0
        Limiting standard deviation.

    Returns
    -------
    RichResult
        keys: ``z``, ``scaling``, ``p_two_sided``, ``bias_subtracted``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 5, Sec. 5.1.3 (asymptotic normality of the density
    estimator).
    """
    from scipy import stats

    n = int(n)
    h = float(h)
    b = float(b)
    if n < 2 or h <= 0 or b <= 0:
        raise ValueError("need n >= 2 and positive h, b.")
    sig = float(sigma)
    if sig <= 0:
        raise ValueError(f"sigma must be positive, got {sig}.")
    scale = np.sqrt(n * h / b)
    z = scale * (float(fn_u) - float(f_u) - float(bias)) / sig
    return RichResult(payload={"z": float(z), "scaling": float(scale),
                               "p_two_sided": float(2 * stats.norm.sf(abs(z))),
                               "bias_subtracted": float(bias),
                               "method": "[n h / b]^{1/2}(f-hat - f - bias) -> N(0, sigma^2)"})


def cheatsheet():
    return "hrzdcnm: b_n is the deconvolution price; the bias is subtracted, not wished away"


#: Catalogue alias for :func:`hrz_deconv_normality`.
horowitz_deconv_normality = hrz_deconv_normality
