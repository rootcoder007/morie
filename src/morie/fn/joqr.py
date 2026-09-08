# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Linear quantile regression.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 17 p. 500
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["quantreg", "joseph_quantile_regression"]

_METHOD = "Linear quantile regression"


def quantreg(x, y, q, iters=25):
    """Linear quantile regression.

    Linear quantile regression, ch. 17 p. 500.

    Fitted by iteratively reweighted least squares on the pinball loss
    with a fixed iteration count and a fixed smoothing floor -- no
    convergence test, so both arms take identical steps.  ``x`` is a
    list of feature rows; an intercept is added.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.quantreg``.
    y : as documented for the shelf core
        See ``morie.fn._joseph.quantreg``.
    q : as documented for the shelf core
        See ``morie.fn._joseph.quantreg``.
    iters : as documented for the shelf core
        See ``morie.fn._joseph.quantreg``.

    Returns
    -------
    result : RichResult
        Payload keys: intercept, loss, q, n.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 17 p. 500
    """
    res = _core.quantreg(x=x, y=y, q=q, iters=iters)
    return RichResult(
        title=_METHOD,
        summary_lines=[("intercept", res["intercept"]), ("loss", res["loss"]), ("q", res["q"]), ("n", res["n"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_quantile_regression = quantreg


def cheatsheet():
    return "quantreg: Linear quantile regression"
