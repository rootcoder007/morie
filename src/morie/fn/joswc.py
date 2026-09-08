# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sliding-window cross-validation.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 5 p. 128
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["slidecv", "joseph_sliding_window_cv"]

_METHOD = "Sliding-window cross-validation"


def slidecv(n, trainsize, testsize, step=None):
    """Sliding-window cross-validation.

    Sliding-window cross-validation, ch. 5 p. 128.

    A fixed-length training window slides forward, so old data drops
    out.  Returns the fold boundaries as half-open [start, end) index
    pairs; the caller fits whatever it likes on them.

    Parameters
    ----------
    n : as documented for the shelf core
        See ``morie.fn._joseph.slidecv``.
    trainsize : as documented for the shelf core
        See ``morie.fn._joseph.slidecv``.
    testsize : as documented for the shelf core
        See ``morie.fn._joseph.slidecv``.
    step : as documented for the shelf core
        See ``morie.fn._joseph.slidecv``.

    Returns
    -------
    result : RichResult
        Payload keys: nfolds, firsttest, lasttest.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 5 p. 128
    """
    res = _core.slidecv(n=n, trainsize=trainsize, testsize=testsize, step=step)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nfolds", res["nfolds"]), ("firsttest", res["firsttest"]), ("lasttest", res["lasttest"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_sliding_window_cv = slidecv


def cheatsheet():
    return "slidecv: Sliding-window cross-validation"
