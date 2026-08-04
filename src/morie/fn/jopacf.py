# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Partial autocorrelation by Durbin-Levinson.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 3
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["pacfts", "joseph_partial_autocorrelation"]

_METHOD = "Partial autocorrelation by Durbin-Levinson"


def pacfts(x, maxlag=20):
    """Partial autocorrelation by Durbin-Levinson.

    Partial autocorrelation by the Durbin-Levinson recursion, ch. 3.

    Fixed recursion depth, no tolerance test, so both arms take
    identical steps.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.pacfts``.
    maxlag : as documented for the shelf core
        See ``morie.fn._joseph.pacfts``.

    Returns
    -------
    result : RichResult
        Payload keys: p1, ci, nsignif, n.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 3
    """
    res = _core.pacfts(x=x, maxlag=maxlag)
    return RichResult(
        title=_METHOD,
        summary_lines=[("p1", res["p1"]), ("ci", res["ci"]), ("nsignif", res["nsignif"]), ("n", res["n"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_partial_autocorrelation = pacfts


def cheatsheet():
    return "pacfts: Partial autocorrelation by Durbin-Levinson"
