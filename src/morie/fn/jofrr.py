# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fourier seasonality terms.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 4 p. 61 and ch. 5 p. 95
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["fourfeat", "joseph_fourier_features"]

_METHOD = "Fourier seasonality terms"


def fourfeat(n, period, k, start=0):
    """Fourier seasonality terms.

    Fourier terms for seasonality, ch. 4 p. 61 and ch. 5 p. 95.

    Column pair j is sin(2 pi j t / m), cos(2 pi j t / m) for
    j = 1..k, evaluated at t = start .. start + n - 1.  The book calls
    these "trigonometric seasonality" (p. 95).

    Parameters
    ----------
    n : as documented for the shelf core
        See ``morie.fn._joseph.fourfeat``.
    period : as documented for the shelf core
        See ``morie.fn._joseph.fourfeat``.
    k : as documented for the shelf core
        See ``morie.fn._joseph.fourfeat``.
    start : as documented for the shelf core
        See ``morie.fn._joseph.fourfeat``.

    Returns
    -------
    result : RichResult
        Payload keys: nrows, ncols, k, sumsq.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 4 p. 61 and ch. 5 p. 95
    """
    res = _core.fourfeat(n=n, period=period, k=k, start=start)
    return RichResult(
        title=_METHOD,
        summary_lines=[("nrows", res["nrows"]), ("ncols", res["ncols"]), ("k", res["k"]), ("sumsq", res["sumsq"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_fourier_features = fourfeat


def cheatsheet():
    return "fourfeat: Fourier seasonality terms"
