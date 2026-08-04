# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Box-Cox transformation.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 164
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["boxcox", "joseph_box_cox_transform"]

_METHOD = "Box-Cox transformation"


def boxcox(x, lam):
    """Box-Cox transformation.

    Box-Cox transformation, ch. 6 p. 164.

    w = (x^lambda - 1) / lambda   for lambda != 0
    w = log(x)                    for lambda == 0

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.boxcox``.
    lam : as documented for the shelf core
        See ``morie.fn._joseph.boxcox``.

    Returns
    -------
    result : RichResult
        Payload keys: lam, mean, var.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 6 p. 164
    """
    res = _core.boxcox(x=x, lam=lam)
    return RichResult(
        title=_METHOD,
        summary_lines=[("lam", res["lam"]), ("mean", res["mean"]), ("var", res["var"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_box_cox_transform = boxcox


def cheatsheet():
    return "boxcox: Box-Cox transformation"
