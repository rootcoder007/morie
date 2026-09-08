# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""Bayes factor between two models from their log marginal likelihoods.

Source: Kass, R. E. and Raftery, A. E. (1995), "Bayes factors", Journal
of the American Statistical Association 90(430), 773-795,
doi:10.1080/01621459.1995.10476572 (citation verified against Crossref).

The Bayes factor is the ratio of marginal likelihoods,

    B_12 = p(D | M1) / p(D | M2),

and the arithmetic is done on the log scale throughout,
log B = log p(D|M1) - log p(D|M2), exponentiating once at the end.  That
is not a style preference: marginal likelihoods routinely underflow a
double, and forming the ratio from the exponentials gives 0/0 for models
that are perfectly well separated on the log scale.  ``log_bf`` is
returned first and ``bf`` may legitimately be ``inf`` or ``0``.

The evidence scale is the paper's own, on the 2 log_e B scale:
0 to 2 "not worth more than a bare mention", 2 to 6 "positive", 6 to 10
"strong", above 10 "very strong".  Note the factor of two -- the same
table is often quoted on the log_10 scale with different cut-points, and
mixing them up moves a result two categories.  Both ``two_log_bf`` and
``log10_bf`` are returned so the reader can see which scale is in play.

The interpretation is symmetric: B_21 = 1/B_12, so evidence for M2 is
read from the same number with the sign of log B flipped.  ``favours``
says which model, and ``category`` grades the strength using |2 log B|.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["bayes_factor"]

_LOG10 = math.log(10.0)


def evidence_category(two_log_bf):
    """Kass and Raftery's grades on the 2 log_e B scale, applied to |2 log B|."""
    a = abs(float(two_log_bf))
    if a < 2.0:
        return "bare mention"
    if a < 6.0:
        return "positive"
    if a < 10.0:
        return "strong"
    return "very strong"


def bayes_factor(log_evidence_1, log_evidence_2):
    """B_12 from two log marginal likelihoods.

    Parameters
    ----------
    log_evidence_1, log_evidence_2 : float
        log p(D | M1) and log p(D | M2).  Logs, not likelihoods.

    Returns
    -------
    bf : B_12, possibly inf or 0
    log_bf : log B_12, always finite for finite inputs
    two_log_bf, log10_bf : the same quantity on the other two scales
    category : the Kass-Raftery grade
    favours : 1 or 2
    """
    l1 = float(log_evidence_1)
    l2 = float(log_evidence_2)
    if l1 != l1 or l2 != l2:
        raise ValueError("bayes_factor: log evidences must not be NaN")
    if math.isinf(l1) and math.isinf(l2) and (l1 > 0) == (l2 > 0):
        raise ValueError("bayes_factor: both log evidences are the same infinity, B is undefined")
    lb = l1 - l2
    if lb > 709.0:
        bf = float("inf")
    elif lb < -745.0:
        bf = 0.0
    else:
        bf = math.exp(lb)
    return RichResult(
        title="Bayes factor",
        summary_lines=[("log_bf", lb), ("category", evidence_category(2.0 * lb))],
        payload={
            "bf": bf,
            "estimate": bf,
            "log_bf": lb,
            "two_log_bf": 2.0 * lb,
            "log10_bf": lb / _LOG10,
            "bf_21": (1.0 / bf) if bf not in (0.0, float("inf")) else (float("inf") if bf == 0.0 else 0.0),
            "category": evidence_category(2.0 * lb),
            "favours": 1 if lb > 0.0 else (2 if lb < 0.0 else 0),
            "log_evidence_1": l1,
            "log_evidence_2": l2,
            "method": "B_12 = exp(log p(D|M1) - log p(D|M2)); Kass and Raftery (1995), 2 log_e scale",
        },
    )


def cheatsheet():
    return "bayfac: Bayes factor between models"


# compact alias per ledger/NAMING.md
bayesfactor = bayes_factor
