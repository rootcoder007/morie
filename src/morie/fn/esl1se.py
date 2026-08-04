# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-standard-error rule for model selection (ESL Ch 7.10)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_one_se_rule"]


def esl_one_se_rule(cv_err, cv_se):
    """
    One-standard-error rule.

    Formula: take the model with minimum CV error, then choose the
    MOST PARSIMONIOUS model whose error is within one standard error
    of that minimum. Parsimony here is index order: the inputs must be
    ordered from simplest to most complex, which is the convention the
    rule depends on and the one thing a caller can silently get wrong,
    so it is stated rather than assumed. Indices are 0-based.

    Parameters
    ----------
    cv_err : array-like
        CV error per candidate, ordered simplest to most complex.
    cv_se : array-like
        Standard error of each CV estimate, same length, >= 0.

    Returns
    -------
    result : dict
        Keys: estimate (chosen index, 0-based), index_min,
        threshold, chosen_error, min_error, n_within, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 7.10 (Fig. 7.9).

    Examples
    --------
    Minimum at index 3, but index 1 is already within one SE:

    >>> out = esl_one_se_rule([0.9, 0.55, 0.52, 0.50], [0.1, 0.1, 0.1, 0.1])
    >>> out["index_min"]
    3
    >>> out["estimate"]
    1
    >>> out["threshold"]
    0.6
    >>> esl_one_se_rule([0.9, 0.8], [0.01, 0.01])["estimate"]
    1
    >>> esl_one_se_rule([0.9, 0.8], [0.01])
    Traceback (most recent call last):
        ...
    ValueError: cv_err (2) and cv_se (1) lengths differ.
    """
    err = np.atleast_1d(np.asarray(cv_err, dtype=float))
    se = np.atleast_1d(np.asarray(cv_se, dtype=float))
    if err.size != se.size:
        raise ValueError(f"cv_err ({err.size}) and cv_se ({se.size}) lengths differ.")
    if err.size == 0:
        raise ValueError("the one-SE rule needs at least one candidate.")
    if np.any(se < 0):
        raise ValueError("standard errors cannot be negative.")
    i_min = int(np.argmin(err))
    threshold = float(err[i_min] + se[i_min])
    within = np.flatnonzero(err <= threshold)
    chosen = int(within[0])
    return RichResult(payload={
        "estimate": chosen, "index_min": i_min, "threshold": threshold,
        "chosen_error": float(err[chosen]), "min_error": float(err[i_min]),
        "n_within": int(within.size),
        "method": "1-SE rule: simplest model within one SE of the CV minimum"})


def cheatsheet():
    return "esl1se: threshold = min + se(min); pick lowest index under it"


# compact alias per ledger/NAMING.md
esloneserule = esl_one_se_rule
