# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seasonal-trend decomposition.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 3 p. 64
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["stldecomp", "joseph_stl_decomposition"]

_METHOD = "Seasonal-trend decomposition"


def stldecomp(x, period, robust=False, iters=2):
    """Seasonal-trend decomposition.

    Seasonal-trend decomposition, ch. 3 p. 64.

    The book's STL uses LOESS smoothers.  This routine uses the
    classical moving-average form of the same three-part model --
    centred moving-average trend, seasonal means of the detrended
    series, remainder -- iterated ``iters`` times.  That substitution
    is OURS and is stated here rather than passed off as STL: the
    LOESS smoother has bandwidth and robustness-iteration choices whose
    defaults differ between implementations, and a decomposition whose
    numbers depend on which library you call cannot be checked across
    two languages.  ``robust`` switches the seasonal aggregate from the
    mean to the median, which is the robustness knob the book
    describes.

    The additive model is x = trend + seasonal + remainder, and the
    seasonal component is centred to sum to zero over one period, as
    STL also does.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.stldecomp``.
    period : as documented for the shelf core
        See ``morie.fn._joseph.stldecomp``.
    robust : as documented for the shelf core
        See ``morie.fn._joseph.stldecomp``.
    iters : as documented for the shelf core
        See ``morie.fn._joseph.stldecomp``.

    Returns
    -------
    result : RichResult
        Payload keys: seasonalstrength, remaindervar, seasonalrange.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 3 p. 64
    """
    res = _core.stldecomp(x=x, period=period, robust=robust, iters=iters)
    return RichResult(
        title=_METHOD,
        summary_lines=[("seasonalstrength", res["seasonalstrength"]), ("remaindervar", res["remaindervar"]), ("seasonalrange", res["seasonalrange"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_stl_decomposition = stldecomp


def cheatsheet():
    return "stldecomp: Seasonal-trend decomposition"
