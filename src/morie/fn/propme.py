# SPDX-License-Identifier: AGPL-3.0-or-later
"""Proportion of the total effect mediated, ab / (c' + ab)."""

from . import _array_core as np

from ._richresult import RichResult
from .propMd import proportion_mediated as _prop_md

__all__ = ["propme"]


def propme(a, b, c_prime, se_a=None, se_b=None, se_c_prime=None):
    """
    Proportion mediated and ratio of mediated to direct effect for the
    single-mediator model M = i1 + a X, Y = i2 + c' X + b M.

    MacKinnon, Warsi and Dwyer (1995) define the proportion of the
    total effect that is mediated as ab / (c' + ab) and the ratio of
    the mediated to the nonmediated effect as ab / c'. The proportion
    itself is the same measure as ``propMd`` (PM = NIE / (NIE + NDE)
    with NIE = ab, NDE = c' under the linear model), and this module
    delegates that core computation to ``propMd.proportion_mediated``
    rather than redefining it. When the path
    standard errors are supplied, first-order (delta-method) variances
    are returned, using the multivariate delta method with independent
    (a, b, c'):

        Var(PM)    = (b^2 c'^2 s_a^2 + a^2 c'^2 s_b^2
                      + a^2 b^2 s_c'^2) / (c' + ab)^4
        Var(ratio) = (b^2 / c'^2) s_a^2 + (a^2 / c'^2) s_b^2
                      + (a^2 b^2 / c'^4) s_c'^2

    which are the first-order (uncorrected) solutions in MacKinnon,
    Warsi and Dwyer (1995).

    Parameters
    ----------
    a : float
        Path X -> M.
    b : float
        Path M -> Y adjusted for X.
    c_prime : float
        Direct effect X -> Y adjusted for M.
    se_a, se_b, se_c_prime : float, optional
        Standard errors of a, b, c'. If all three are given, delta
        method standard errors of the two measures are returned.

    Returns
    -------
    result : RichResult
        Keys: estimate (proportion mediated), same_sign (from propMd),
        ratio, indirect (ab), total (c' + ab), and, when the ses are
        supplied, se and se_ratio.

    References
    ----------
    MacKinnon, D. P., Warsi, G. and Dwyer, J. H. (1995), "A simulation
    study of mediated effect measures", Multivariate Behavioral
    Research 30(1), 41-62, doi:10.1207/s15327906mbr3001_3; proportion
    mediated ab / (c' + ab), ratio ab / c', and their first-order
    variance solutions. Full text verified at
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2821114/ (PMC blocks PDF
    download; formulas confirmed from the article body).
    """
    a = float(a)
    b = float(b)
    c_prime = float(c_prime)
    ab = a * b
    total = c_prime + ab
    if total == 0.0:
        raise ValueError("total effect c_prime + a*b is zero")
    core = _prop_md(ab, c_prime)
    payload = {
        "estimate": float(core["estimate"]),
        "same_sign": float(core["same_sign"]),
        "indirect": ab,
        "total": total,
        "ratio": ab / c_prime if c_prime != 0.0 else np.nan,
        "method": "MacKinnon-Warsi-Dwyer (1995) proportion mediated",
    }
    if se_a is not None and se_b is not None and se_c_prime is not None:
        sa2 = float(se_a) ** 2
        sb2 = float(se_b) ** 2
        sc2 = float(se_c_prime) ** 2
        var_pm = (b * b * c_prime * c_prime * sa2
                  + a * a * c_prime * c_prime * sb2
                  + a * a * b * b * sc2) / total ** 4
        payload["se"] = float(np.sqrt(var_pm))
        if c_prime != 0.0:
            var_r = ((b * b / c_prime ** 2) * sa2
                     + (a * a / c_prime ** 2) * sb2
                     + (a * a * b * b / c_prime ** 4) * sc2)
            payload["se_ratio"] = float(np.sqrt(var_r))
        else:
            payload["se_ratio"] = np.nan
    return RichResult(payload=payload)


def cheatsheet():
    return "propme(a, b, c_prime) -> proportion mediated ab / (c_prime + ab) and mediated-to-direct ratio."
