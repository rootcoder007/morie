# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Winkler interval score.

Winkler (1972) JASA 67(337):187-191; Gneiting and Raftery (2007) JASA 102(477) eq. (43) -- NOT printed in Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["winkler", "joseph_winkler_interval_score"]

_METHOD = "Winkler interval score"


def winkler(y, lower, upper, alpha=0.1):
    """Winkler interval score.

    Winkler interval score, ch. 17.

    NOT LOCATED IN THE EXTRACTED TEXT: the corpus copy of Joseph and
    Tackes never prints the Winkler score, so it is taken from the
    primary source and stated here in full:

        W = (u - l)
            + (2/alpha)(l - y)  if y < l
            + (2/alpha)(y - u)  if y > u

    -- Winkler, R. L. (1972), "A Decision-Theoretic Approach to
    Interval Estimation", Journal of the American Statistical
    Association 67(337):187-191; in the form popularized by Gneiting,
    T. and Raftery, A. E. (2007), "Strictly Proper Scoring Rules,
    Prediction, and Estimation", JASA 102(477):359-378, eq. (43).
    Lower is better.

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._joseph.winkler``.
    lower : as documented for the shelf core
        See ``morie.fn._joseph.winkler``.
    upper : as documented for the shelf core
        See ``morie.fn._joseph.winkler``.
    alpha : as documented for the shelf core
        See ``morie.fn._joseph.winkler``.

    Returns
    -------
    result : RichResult
        Payload keys: score, coverage, meanwidth.

    References
    ----------
    Winkler (1972) JASA 67(337):187-191; Gneiting and Raftery (2007) JASA 102(477) eq. (43) -- NOT printed in Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt
    """
    res = _core.winkler(y=y, lower=lower, upper=upper, alpha=alpha)
    return RichResult(
        title=_METHOD,
        summary_lines=[("score", res["score"]), ("coverage", res["coverage"]), ("meanwidth", res["meanwidth"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_winkler_interval_score = winkler


def cheatsheet():
    return "winkler: Winkler interval score"
