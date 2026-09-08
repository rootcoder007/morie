# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DirRec multi-step strategy.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 18 p. 551
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["dirrec", "joseph_dirrec_strategy"]

_METHOD = "DirRec multi-step strategy"


def dirrec(x, lags, horizon):
    """DirRec multi-step strategy.

    DirRec strategy, ch. 18 p. 551.

    The hybrid the book names: like Direct, a separate model per
    horizon; like Recursive, each successive model may also use the
    forecasts already produced, so the input space GROWS by one column
    at every step.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.dirrec``.
    lags : as documented for the shelf core
        See ``morie.fn._joseph.dirrec``.
    horizon : as documented for the shelf core
        See ``morie.fn._joseph.dirrec``.

    Returns
    -------
    result : RichResult
        Payload keys: first, last, ncolsfirst, ncolslast.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 18 p. 551
    """
    res = _core.dirrec(x=x, lags=lags, horizon=horizon)
    return RichResult(
        title=_METHOD,
        summary_lines=[("first", res["first"]), ("last", res["last"]), ("ncolsfirst", res["ncolsfirst"]), ("ncolslast", res["ncolslast"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_dirrec_strategy = dirrec


def cheatsheet():
    return "dirrec: DirRec multi-step strategy"
