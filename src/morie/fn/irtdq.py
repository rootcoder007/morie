# morie.fn -- function file (rootcoder007/morie)
"""Quadratic-utility IRT vote probability."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["irt_quadratic_utility"]


def irt_quadratic_utility(ideal_point, yea_position, nay_position, noise_sd=1.0):
    r"""Vote probability implied by quadratic utility + normal errors.

    With :math:`U_i(y) = -(x_i - y)^2 + \eta` and independent normal
    errors, the yea probability reduces to the two-parameter probit

    .. math:: P(\text{yea}) = \Phi\big(\alpha_j (x_i - m_j)\big),
              \qquad m_j = \tfrac{z_y + z_n}{2},\;
              \alpha_j = \tfrac{2 (z_y - z_n)}{\sigma\sqrt2},

    i.e. the midpoint :math:`m_j` is the indifference point and the
    discrimination is proportional to how far apart the alternatives
    are -- the quadratic-utility foundation of the CJR item-response
    model, and the contrast with NOMINATE's Gaussian utility.

    Parameters
    ----------
    ideal_point : scalar or array-like
        Ideal point(s) x_i.
    yea_position, nay_position : float
        Outcome locations z_y, z_n.
    noise_sd : float, default 1.0
        Std dev of the utility-error difference.

    Returns
    -------
    RichResult
        keys: ``p_yea`` (same shape as ideal_point), ``midpoint``,
        ``discrimination`` (signed: negative when yea is above nay),
        ``method``.

    References
    ----------
    Clinton, J., Jackman, S. & Rivers, D. (2004). The statistical
    analysis of roll call data. *APSR*, 98(2), 355-370. (quadratic
    utility -> probit IRT)
    """
    x = np.asarray(ideal_point, dtype=float)
    zy = float(yea_position)
    zn = float(nay_position)
    s = float(noise_sd)
    if s <= 0:
        raise ValueError(f"noise_sd must be positive, got {s}.")
    if zy == zn:
        raise ValueError("yea and nay positions coincide; the vote is uninformative.")

    mid = (zy + zn) / 2.0
    # U_yea - U_nay = -(x - z_y)^2 + (x - z_n)^2 = 2 (z_y - z_n)(x - mid)
    disc = 2.0 * (zy - zn) / (s * np.sqrt(2.0))
    p = stats.norm.cdf(disc * (x - mid))

    return RichResult(
        payload={
            "p_yea": float(p) if np.ndim(ideal_point) == 0 else p,
            "midpoint": mid,
            "discrimination": disc,
            "method": "Quadratic-utility probit IRT vote probability (CJR 2004)",
        }
    )


def cheatsheet():
    return "irtdq: P(yea) = Phi(alpha (x - midpoint)); alpha prop. to outcome separation"
