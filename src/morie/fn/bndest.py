# morie.fn -- function file (rootcoder007/morie)
"""Manski worst-case bounds on a partially observed mean."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bound_estimation"]


def bound_estimation(y, observed, support, treatment=None):
    r"""Worst-case (no-assumption) bounds on a mean with missing
    outcomes, and on an average treatment effect with unobserved
    counterfactuals (Manski 1989, 1990).

    With :math:`P(\text{obs})` the probability the outcome is seen
    and :math:`[K_0, K_1]` its logical support,

    .. math:: E[Y] \in \big[\,E[Y\mid \text{obs}]P(\text{obs})
              + K_0(1 - P(\text{obs})),\;
              E[Y\mid \text{obs}]P(\text{obs})
              + K_1(1 - P(\text{obs}))\,\big].

    The width is exactly :math:`(K_1 - K_0)(1 - P(\text{obs}))` --
    the support width times the missing share -- and that identity
    is tested rather than assumed. Nothing about the missing data is
    assumed at all; that is the point, and it is also why the bounds
    are wide. They are what the DATA alone say, before any
    missing-at-random or selection assumption is spent.

    With ``treatment`` supplied, the same construction bounds each
    potential-outcome mean -- every treated unit's untreated outcome
    is missing, and vice versa -- and the ATE bounds difference them:
    lower(ATE) = lower(E[Y(1)]) - upper(E[Y(0)]). The ATE interval
    always has width :math:`K_1 - K_0` exactly and therefore ALWAYS
    contains zero: worst-case bounds never sign a treatment effect on
    their own. An implementation whose no-assumption ATE bounds
    exclude zero has smuggled in an assumption, and the test asserts
    this cannot happen.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome; entries where ``observed`` is False may hold
        anything (they are ignored).
    observed : array-like of bool, shape (n,)
        Whether the outcome was seen.
    support : (float, float)
        The logical range :math:`[K_0, K_1]` of the outcome. This is
        an ASSUMPTION about the outcome's meaning (a share lies in
        [0, 1], a test score in [0, 100]) and the only one used.
    treatment : array-like of 0/1, optional
        When supplied, ``observed`` is ignored and the missingness is
        the counterfactual one: Y(1) is seen only for treated units.

    Returns
    -------
    RichResult
        keys: ``lower``, ``upper``, ``width``, ``p_observed``,
        ``identified`` (False unless nothing is missing), ``ate_*``
        keys when ``treatment`` is given, ``contains_zero``, ``n``,
        ``method``.

    References
    ----------
    Manski, C. F. (1989), "Anatomy of the selection problem",
    *Journal of Human Resources* 24:343-360. Manski, C. F. (1990),
    "Nonparametric bounds on treatment effects", *American Economic
    Review Papers and Proceedings* 80:319-323. Manski, C. F. and
    Tamer, E. (2002), *Econometrica* 70:519-546, for interval data.
    """
    yv = np.asarray(y, dtype=float).ravel()
    k0, k1 = float(support[0]), float(support[1])
    if not k0 < k1:
        raise ValueError(f"the support must satisfy K0 < K1, got {support}.")
    n = yv.size

    def one_mean(seen):
        p = float(np.mean(seen))
        if p > 0:
            ys = yv[seen]
            if np.any(ys < k0 - 1e-12) or np.any(ys > k1 + 1e-12):
                raise ValueError(
                    "an observed outcome lies outside the declared support; "
                    "the support is the one assumption here, so violating "
                    "it voids the bounds.")
            m = float(ys.mean())
        else:
            m = 0.0
        return (m * p + k0 * (1 - p), m * p + k1 * (1 - p), p)

    if treatment is None:
        obs = np.asarray(observed, dtype=bool).ravel()
        if obs.size != n:
            raise ValueError(f"observed has {obs.size} entries for {n}.")
        lo, hi, p = one_mean(obs)
        return RichResult(payload={
            "lower": lo, "upper": hi, "width": hi - lo,
            "p_observed": p, "identified": bool(p == 1.0),
            "width_identity": "(K1 - K0)(1 - P(obs)) exactly",
            "assumptions": "the outcome's support alone; nothing about "
                           "WHY data are missing",
            "n": int(n),
            "method": "Manski worst-case bounds on a partially observed mean"})

    Tv = np.asarray(treatment, dtype=float).ravel()
    if Tv.size != n:
        raise ValueError(f"treatment has {Tv.size} entries for {n}.")
    if not np.all(np.isin(Tv, (0.0, 1.0))):
        raise ValueError("treatment must be binary 0/1.")
    lo1, hi1, p1 = one_mean(Tv == 1)
    lo0, hi0, p0 = one_mean(Tv == 0)
    ate_lo = lo1 - hi0
    ate_hi = hi1 - lo0
    return RichResult(payload={
        "ate_lower": ate_lo, "ate_upper": ate_hi,
        "ate_width": ate_hi - ate_lo,
        "y1_bounds": (lo1, hi1), "y0_bounds": (lo0, hi0),
        "p_treated": p1, "contains_zero": bool(ate_lo <= 0.0 <= ate_hi),
        "width_identity": "the ATE bounds always have width exactly K1 - K0, "
                          "so they always contain zero: no-assumption bounds "
                          "never sign an effect on their own",
        "identified": False,
        "n": int(n),
        "method": "Manski (1990) worst-case bounds on the average treatment effect"})


def cheatsheet():
    return "bndest: no-assumption ATE bounds have width K1 - K0 and always contain zero"
