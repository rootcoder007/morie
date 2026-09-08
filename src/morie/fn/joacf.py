# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sample autocorrelation function.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 3
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["autocorf", "joseph_autocorrelation_function"]

_METHOD = "Sample autocorrelation function"


def autocorf(x, maxlag=20):
    """Sample autocorrelation function.

    Autocorrelation function, ch. 3.

    The standard biased (divide-by-n) estimator, which is what the
    book's ACF plots use:

        r_k = sum_{t=k+1..n} (x_t - xbar)(x_{t-k} - xbar)
              / sum_{t=1..n} (x_t - xbar)^2

    ``ci`` is the +/- 1.96/sqrt(n) band drawn on those plots.

    Parameters
    ----------
    x : as documented for the shelf core
        See ``morie.fn._joseph.autocorf``.
    maxlag : as documented for the shelf core
        See ``morie.fn._joseph.autocorf``.

    Returns
    -------
    result : RichResult
        Payload keys: r1, ci, nsignif, n.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 3
    """
    res = _core.autocorf(x=x, maxlag=maxlag)
    return RichResult(
        title=_METHOD,
        summary_lines=[("r1", res["r1"]), ("ci", res["ci"]), ("nsignif", res["nsignif"]), ("n", res["n"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_autocorrelation_function = autocorf


def cheatsheet():
    return "autocorf: Sample autocorrelation function"
