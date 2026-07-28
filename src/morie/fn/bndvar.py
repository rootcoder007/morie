# morie.fn -- function file (rootcoder007/morie)
"""Confidence intervals for a partially identified parameter."""

import numpy as np

from ._richresult import RichResult

__all__ = ["bound_variance_term"]


def bound_variance_term(lower_hat, upper_hat, se_lower, se_upper, n,
                        alpha=0.05):
    r"""The Imbens-Manski confidence interval for a parameter that is
    only known to lie in :math:`[\theta_l, \theta_u]`:

    .. math:: \big[\hat\theta_l - c_n\,\hat\sigma_l/\sqrt n,\;
              \hat\theta_u + c_n\,\hat\sigma_u/\sqrt n\big],

    with :math:`c_n` solving

    .. math:: \Phi\!\left(c_n + \sqrt n\,\hat\Delta/
              \max(\hat\sigma_l, \hat\sigma_u)\right)
              - \Phi(-c_n) = 1 - \alpha,
              \qquad \hat\Delta = \hat\theta_u - \hat\theta_l .

    The subtlety this construction exists for: a confidence interval
    can cover the identified SET or the true PARAMETER, and they are
    different targets. Covering the set needs the two-sided
    :math:`z_{1-\alpha/2}`; covering the parameter needs only the
    one-sided :math:`z_{1-\alpha}` when the interval is wide, because
    the parameter can be missed at one end at a time. :math:`c_n`
    interpolates: it equals :math:`z_{1-\alpha}` when
    :math:`\hat\Delta` is large relative to sampling noise and rises
    to :math:`z_{1-\alpha/2}` as the interval narrows to a point --
    at which point the problem is point identification and the usual
    two-sided interval is what comes out. Both limits are tested.
    Stoye (2009) showed the interpolation needs :math:`\hat\Delta`'s
    own sampling error to be negligible or the bound superefficient;
    the output says so rather than hiding the caveat.

    Parameters
    ----------
    lower_hat, upper_hat : float
        Estimated bounds.
    se_lower, se_upper : float
        Their standard deviations, on the :math:`\sqrt n` scale --
        i.e. ``sqrt(n) * se(bound)``.
    n : int
        Sample size.
    alpha : float, default 0.05
        Miss probability.

    Returns
    -------
    RichResult
        keys: ``ci``, ``c``, ``z_one_sided``, ``z_two_sided``,
        ``delta``, ``covers``, ``stoye_caveat``, ``n``, ``method``.

    References
    ----------
    Imbens, G. W. and Manski, C. F. (2004), "Confidence intervals for
    partially identified parameters", *Econometrica* 72:1845-1857,
    Eq. (6). Stoye, J. (2009), "More on confidence intervals for
    partially identified parameters", *Econometrica* 77:1299-1315.
    """
    from scipy import optimize, stats

    tl, tu = float(lower_hat), float(upper_hat)
    sl, su = float(se_lower), float(se_upper)
    nn = int(n)
    a = float(alpha)
    if tu < tl:
        raise ValueError(
            f"upper_hat must be at least lower_hat, got [{tl}, {tu}].")
    if sl <= 0 or su <= 0:
        raise ValueError("both standard deviations must be positive.")
    if nn < 2:
        raise ValueError(f"n must be at least 2, got {nn}.")
    if not 0 < a < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {a}.")
    delta = tu - tl
    smax = max(sl, su)
    shift = np.sqrt(nn) * delta / smax
    z1 = float(stats.norm.ppf(1 - a))
    z2 = float(stats.norm.ppf(1 - a / 2))

    def gap(c):
        return stats.norm.cdf(c + shift) - stats.norm.cdf(-c) - (1 - a)

    # c is monotone in gap; bracket by the two z's
    if gap(z1) >= 0:
        c = z1
    elif gap(z2) <= 0:
        c = z2
    else:
        c = float(optimize.brentq(gap, z1, z2, xtol=1e-12))
    ci = (tl - c * sl / np.sqrt(nn), tu + c * su / np.sqrt(nn))
    return RichResult(payload={
        "ci": ci, "c": c, "z_one_sided": z1, "z_two_sided": z2,
        "delta": delta,
        "covers": "the TRUE PARAMETER at 1 - alpha; a set-covering interval "
                  "would use the two-sided z throughout and be wider",
        "interpolation": "c equals the one-sided z when the identified set "
                         "is wide relative to noise and rises to the "
                         "two-sided z as it collapses to a point",
        "stoye_caveat": "the interpolation presumes delta_hat's own sampling "
                        "error is negligible or the bound superefficient "
                        "(Stoye 2009); with a noisy delta_hat use his "
                        "modified construction",
        "n": nn,
        "method": "Imbens-Manski (2004) Eq. (6) confidence interval for a "
                  "partially identified parameter"})


def cheatsheet():
    return "bndvar: covering the PARAMETER needs c between z_{1-a} and z_{1-a/2} -- Imbens-Manski interpolates"
