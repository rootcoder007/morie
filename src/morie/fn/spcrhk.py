# morie.fn -- function file (rootcoder007/morie)
"""Cressie-Hawkins robust semivariogram estimator."""

from . import _array_core as np

from ._richresult import RichResult
from ._schaben import cressie_hawkins, matheron

__all__ = ["schabenberger_cressie_hawkins"]


def schabenberger_cressie_hawkins(coords, z, lag_bins=None, cutoff=None,
                                  exact=False, full_correction=False):
    r"""The robust semivariogram estimator, Schabenberger eq (4.26).

    .. math::
       \hat\gamma_{CH}(h) = \frac{1}{2}
       \left\{\frac{1}{|N(h)|}\sum_{N(h)}|Z(s_i)-Z(s_j)|^{1/2}\right\}^{4}
       \Big/\left(0.457 + \frac{0.494}{|N(h)|}\right)

    The square-root differences are averaged FIRST and the average is
    then raised to the fourth power, which is what stops a single
    extreme value dominating; the denominator restores approximate
    unbiasedness.

    Two cautions the book is explicit about. Robust here means
    resistant to slight contamination of a Gaussian field, not
    resistant in general: the influence function is unbounded and the
    breakdown point is 0 %. And the robustness is not free -- at
    :math:`m = 1` the two estimators vary about equally and beyond that
    the robust one is MORE variable, which Hawkins and Cressie describe
    as "a premium paid by the robust estimators on normal data to
    insure against the effects of possible outliers".

    The bias correction is derived on p. 160 as
    :math:`0.457 + 0.494/|N(h)| + 0.045/|N(h)|^2`, and the last term is
    then dropped when equation (4.26) is written down. The printed
    equation is the default here because it is what the book's own
    worked Example 4.3 evaluates -- its factor 0.704 at
    :math:`|N(h)|=2` is :math:`0.457+0.494/2` with no
    :math:`0.045/4` term. ``full_correction=True`` restores it.

    Parameters
    ----------
    coords, z, lag_bins, cutoff, exact
        As in :func:`~morie.fn.spmath.schabenberger_matheron_estimator`.
    full_correction : bool
        Include the :math:`0.045/|N(h)|^2` term.

    Returns
    -------
    RichResult
        ``gamma``, ``lag``, ``n_pairs``, ``matheron`` (the classical
        estimator on the same lags, for comparison), ``ratio``.

    References
    ----------
    Schabenberger and Gotway (2005), section 4.4.2, equation (4.26),
    pp. 159-161. Cressie and Hawkins (1980), *Journal of the
    International Association for Mathematical Geology* 12:115-125.
    Hawkins and Cressie (1984).

    Examples
    --------
    >>> import numpy as np
    >>> co = np.array([[1, 1], [1, 4], [2, 2], [3, 1], [3, 4]], float)
    >>> z = np.array([1, 4, 2, 3, 20], float)
    >>> out = schabenberger_cressie_hawkins(co, z, exact=True)
    >>> [round(float(g), 2) for g in out["gamma"]]
    [0.71, 38.14, 45.45, 52.17, 36.61]
    """
    lag, gam, npair = cressie_hawkins(coords, z, lag_bins, cutoff, exact)
    if full_correction:
        base = gam * (0.457 + 0.494 / npair)
        gam = base / (0.457 + 0.494 / npair + 0.045 / npair ** 2)
    _, mat, _, _ = matheron(coords, z, lag_bins, cutoff, exact)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mat > 0, gam / mat, np.nan)
    return RichResult(
        payload={
            "estimate": gam,
            "robust_variogram": gam,
            "gamma": gam,
            "lag": lag,
            "n_pairs": npair,
            "matheron": mat,
            "ratio": ratio,
            "ratio_note": (
                "gamma_CH / gamma_Matheron; well below 1 at a lag means the "
                "classical estimator is being inflated by extreme "
                "differences at that separation"
            ),
            "full_correction": bool(full_correction),
            "correction_note": (
                "the 0.045/|N(h)|^2 term is included" if full_correction else
                "equation (4.26) as printed, which omits the "
                "0.045/|N(h)|^2 term the derivation on p. 160 carries"
            ),
            "robustness_note": (
                "resistant to slight contamination of a Gaussian field only; "
                "the influence function is unbounded and the breakdown point "
                "is zero, and on clean data this estimator is the more "
                "variable of the two"
            ),
            "n": int(np.asarray(z).size),
            "method": "Cressie-Hawkins robust semivariogram estimator",
        }
    )


def cheatsheet():
    return (
        "spcrhk: the fourth-power-of-mean-root robust semivariogram (4.26), "
        "returned alongside Matheron so the contamination is visible"
    )
