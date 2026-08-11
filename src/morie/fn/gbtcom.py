# morie.fn -- wave3 slice w5_08 (rootcoder007/morie)
"""Goodman-Bacon three-way composition of the TWFE DiD coefficient."""

from . import _array_core as np

from .gbacon import goodman_bacon_decomp
from ._richresult import RichResult

__all__ = ["gbtcom", "goodman_bacon_3way"]

_TYPES = (
    "treated vs never-treated",
    "early vs late (before late adopts)",
    "late vs early (early already treated)",
)


def gbtcom(y, D, unit, time):
    r"""Three-way composition of the TWFE coefficient by comparison type.

    Goodman-Bacon's Theorem 1 writes the two-way fixed-effects DiD
    coefficient as a weighted average over every timing-group 2x2:

    .. math:: \hat\beta^{DD} = \sum_{k \neq U} s_{kU}\,
              \hat\beta^{2\times2}_{kU}
              + \sum_{k \neq U}\sum_{\ell > k}
              \big[ s_{k\ell}\,\mu_{k\ell}\, \hat\beta^{2\times2,k}_{k\ell}
              + s_{k\ell}\,(1-\mu_{k\ell})\,
                \hat\beta^{2\times2,\ell}_{k\ell} \big].

    Exactly three KINDS of 2x2 appear (eqs. 7-9 of the source): a
    timing cohort against the never-treated units, an earlier cohort
    against a later one before the later adopts, and a later cohort
    against an earlier one that is ALREADY treated -- the forbidden
    comparison that differences out the earlier cohort's evolving
    effect. This function collapses the full decomposition
    (:func:`morie.fn.gbacon.goodman_bacon_decomp`) to that three-way
    composition: total weight, weighted-average 2x2 estimate and
    weight-times-estimate contribution per type, so the reader can see
    at a glance how much of a TWFE coefficient rests on each kind of
    variation. The composition is an identity: the three
    contributions sum back to the TWFE coefficient, and the three
    weights sum to 1; both are computed and reported as
    ``identity_residual`` and ``weight_sum`` rather than assumed.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome in long format.
    D : array-like, shape (n,)
        Absorbing binary treatment.
    unit, time : array-like, shape (n,)
        Identifiers; the panel must be balanced.

    Returns
    -------
    RichResult
        ``estimate`` (the TWFE coefficient), ``weight`` (dict, total
        weight per type), ``mean_beta`` (dict, weighted-average 2x2
        estimate per type), ``contribution`` (dict, weight x mean),
        ``weight_sum``, ``identity_residual``, ``forbidden_weight``,
        ``n_components_by_type``.

    References
    ----------
    Goodman-Bacon, A. (2021), "Difference-in-differences with
    variation in treatment timing", Journal of Econometrics
    225(2):254-277, doi:10.1016/j.jeconom.2021.03.014. Implemented
    from Theorem 1 and eqs. (7)-(9) of the NBER Working Paper 25018
    version, local copy
    /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    goodman-bacon-2021-did-variation-treatment-timing.pdf.

    Examples
    --------
    >>> import numpy as np
    >>> unit = np.repeat(np.arange(9), 8)
    >>> time = np.tile(np.arange(1, 9), 9)
    >>> g = np.repeat([3., 3., 3., 5., 5., 5., np.inf, np.inf, np.inf], 8)
    >>> rel = np.where(np.isfinite(g), np.maximum(0, time - g), 0.)
    >>> y = np.repeat(np.arange(9), 8) * 0.3 + time * 0.2 + \
    ...     np.where(time >= g, 1 + 0.5 * rel, 0.)
    >>> out = gbtcom(y, (time >= g).astype(float), unit, time)
    >>> round(out["weight"]["treated vs never-treated"], 10)
    0.7
    >>> round(abs(out["identity_residual"]), 12)
    0.0
    """
    dec = goodman_bacon_decomp(y, D, unit, time)
    w = {t: 0.0 for t in _TYPES}
    wb = {t: 0.0 for t in _TYPES}
    n_by = {t: 0 for t in _TYPES}
    for c in dec["components"]:
        t = c["type"]
        w[t] += c["weight"]
        wb[t] += c["weight"] * c["beta"]
        n_by[t] += 1
    mean_beta = {t: (wb[t] / w[t]) if w[t] > 0 else np.nan for t in _TYPES}
    wsum = float(sum(w.values()))
    recomposed = float(sum(wb.values()))
    beta = float(dec["estimate"])
    return RichResult(
        payload={
            "estimate": beta,
            "weight": w,
            "mean_beta": mean_beta,
            "contribution": wb,
            "weight_sum": wsum,
            "identity_residual": beta - recomposed,
            "forbidden_weight": float(w[_TYPES[2]]),
            "n_components_by_type": n_by,
            "reading": (
                "the three contributions sum to the TWFE coefficient; a "
                "large forbidden_weight with dynamic effects is how TWFE "
                "lands below every clean comparison"
            ),
            "method": (
                "Goodman-Bacon (2021) three-way composition of the TWFE "
                "DiD coefficient"
            ),
        }
    )


# stub-era long name kept as alias.
goodman_bacon_3way = gbtcom


def cheatsheet():
    return (
        "gbtcom: collapse the Goodman-Bacon decomposition to its three "
        "comparison types (never-treated / early-vs-late / forbidden "
        "late-vs-early); contributions sum back to the TWFE coefficient"
    )
