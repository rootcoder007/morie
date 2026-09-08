# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pinball (quantile) loss.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 17 p. 494; formula from TFT eq. (25), arXiv:1912.09363
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["pinball", "joseph_pinball_quantile_loss"]

_METHOD = "Pinball (quantile) loss"


def pinball(y, qhat, q):
    """Pinball (quantile) loss.

    Pinball (quantile) loss, ch. 17 p. 494.

    The book points at the quantile loss for probabilistic forecasts
    ("we can use quantile loss or pinball loss", p. 494).  The
    canonical statement is TFT eq. (25), which the book's own
    architecture chapter builds on:

        QL(y, yhat, q) = q (y - yhat)_+ + (1 - q) (yhat - y)_+

    -- Lim, B., Arik, S. O., Loeff, N. and Pfister, T., "Temporal
    Fusion Transformers for Interpretable Multi-horizon Time Series
    Forecasting", arXiv:1912.09363, eq. (25).

    Parameters
    ----------
    y : as documented for the shelf core
        See ``morie.fn._joseph.pinball``.
    qhat : as documented for the shelf core
        See ``morie.fn._joseph.pinball``.
    q : as documented for the shelf core
        See ``morie.fn._joseph.pinball``.

    Returns
    -------
    result : RichResult
        Payload keys: loss, total, coverage.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 17 p. 494; formula from TFT eq. (25), arXiv:1912.09363
    """
    res = _core.pinball(y=y, qhat=qhat, q=q)
    return RichResult(
        title=_METHOD,
        summary_lines=[("loss", res["loss"]), ("total", res["total"]), ("coverage", res["coverage"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_pinball_quantile_loss = pinball


def cheatsheet():
    return "pinball: Pinball (quantile) loss"
