# morie.fn -- function file (rootcoder007/morie)
"""Propensity-score nearest-neighbour 1:1 matching."""

import numpy as np

from ._richresult import RichResult
from .cipsc import caliper_psm

__all__ = ["causal_pair_matching"]


def causal_pair_matching(ps, treat, caliper=None, y=None):
    """Nearest-neighbour 1:1 propensity matching (optionally calipered).

    Greedy matching without replacement of each treated unit to the
    closest control on the logit propensity scale. With ``caliper=None``
    every treated unit is matched to its nearest control regardless of
    distance (classical nearest-neighbour matching); pass a caliper --
    or use :func:`morie.fn.cipsc.caliper_psm` directly with its Austin
    (2011) default -- to discard poor matches.

    Parameters
    ----------
    ps : array-like, shape (n,)
        Estimated propensity scores in (0, 1).
    treat : array-like of {0, 1}, shape (n,)
        Treatment indicator.
    caliper : float, optional
        Logit-scale caliper; None matches everyone.
    y : array-like, optional
        Outcome; if given, the matched-pair ATT is reported.

    Returns
    -------
    RichResult
        Same keys as :func:`caliper_psm`: ``matched_idx``,
        ``n_matched``, ``n_treated``, ``caliper``, ``balance``,
        ``att``, ``method``.

    References
    ----------
    Rubin, D. B. (1973). Matching to remove bias in observational
    studies. *Biometrics*, 29(1), 159-183.

    Rosenbaum, P. R. & Rubin, D. B. (1985). Constructing a control
    group using multivariate matched sampling methods that incorporate
    the propensity score. *The American Statistician*, 39(1), 33-38.
    """
    if caliper is None:
        caliper = np.inf  # ponytail: NN matching = caliper matching with infinite caliper
    out = caliper_psm(ps, treat, caliper=caliper, y=y)
    out["method"] = "Propensity-score nearest-neighbour 1:1 matching"
    return out


def cheatsheet():
    return "causmtch: greedy 1:1 NN propensity matching (cipsc with infinite caliper)"
