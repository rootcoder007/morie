# morie.fn -- function file (rootcoder007/morie)
"""Weak identification check for mediation."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["weak_identification_mediation"]


def weak_identification_mediation(a, b, se_a, se_b, threshold=3.0):
    r"""Flag a mediation estimate whose paths are too weak to trust.

    Both the a-path (X -> M) and the b-path (M -> Y | X) must be
    clearly nonzero for :math:`ab` to be interpretable: when either is
    weak the product is near zero with a variance dominated by noise,
    and normal-theory intervals for :math:`ab` are badly behaved
    (Sobel's delta-method standard error assumes both paths are well
    identified). The check reports each path's z statistic, flags any
    below ``threshold``, and gives the Sobel standard error

    .. math:: \mathrm{SE}_{ab} = \sqrt{b^2 s_a^2 + a^2 s_b^2}

    only as a reference -- with a weak path it understates the true
    uncertainty, which is why bootstrap intervals are preferred.

    Parameters
    ----------
    a, b : float
        Path estimates.
    se_a, se_b : float
        Their standard errors (positive).
    threshold : float, default 3.0
        Minimum |z| for a path to count as well identified.

    Returns
    -------
    RichResult
        keys: ``z_a``, ``z_b``, ``weak_a``, ``weak_b``,
        ``weakly_identified``, ``ab``, ``sobel_se``, ``sobel_z``,
        ``sobel_p``, ``threshold``, ``method``.

    References
    ----------
    MacKinnon, D. P., Lockwood, C. M., Hoffman, J. M., West, S. G. &
    Sheets, V. (2002). A comparison of methods to test mediation and
    other intervening variable effects. *Psychological Methods*, 7(1),
    83-104. (the normal-theory product test's poor behaviour)

    Sobel, M. E. (1982). Asymptotic confidence intervals for indirect
    effects in structural equation models. *Sociological Methodology*,
    13, 290-312.
    """
    a, b = float(a), float(b)
    se_a, se_b = float(se_a), float(se_b)
    if se_a <= 0 or se_b <= 0:
        raise ValueError("standard errors must be positive.")
    thr = float(threshold)
    if thr <= 0:
        raise ValueError(f"threshold must be positive, got {thr}.")

    za, zb = a / se_a, b / se_b
    ab = a * b
    se_ab = float(np.sqrt(b**2 * se_a**2 + a**2 * se_b**2))
    z_ab = ab / se_ab if se_ab > 0 else float("nan")
    p_ab = float(2 * stats.norm.sf(abs(z_ab))) if se_ab > 0 else float("nan")

    return RichResult(
        payload={
            "z_a": za,
            "z_b": zb,
            "weak_a": bool(abs(za) < thr),
            "weak_b": bool(abs(zb) < thr),
            "weakly_identified": bool(abs(za) < thr or abs(zb) < thr),
            "ab": ab,
            "sobel_se": se_ab,
            "sobel_z": float(z_ab),
            "sobel_p": p_ab,
            "threshold": thr,
            "method": "Weak-path check for mediation (both |z| >= threshold) + Sobel reference",
        }
    )


def cheatsheet():
    return "weakid: flag |z_a| or |z_b| < threshold; Sobel SE = sqrt(b^2 sa^2 + a^2 sb^2)"
