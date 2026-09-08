# morie.fn -- function file (rootcoder007/morie)
"""Phillips-Perron unit-root test (trend-argument spelling)."""

from __future__ import annotations

from .pptest import phillips_perron_unit_root

__all__ = ["phillips_perron"]


def phillips_perron(y, trend=True, lags=None, kind="Z(t_alpha)"):
    """Phillips-Perron unit-root test with an explicit trend switch.

    The implementation lives in
    :func:`morie.fn.pptest.phillips_perron_unit_root`, whose auxiliary
    regression always includes the linear trend -- that is the
    ``tseries::pp.test`` specification and the one the tabulated
    critical values belong to.  ``trend=False`` is therefore refused
    rather than silently answered with trend-case critical values, which
    would be the wrong table and a p-value that looks fine.

    Parameters
    ----------
    y : array-like
        Series in time order.
    trend : bool
        Must be ``True``; see above.
    lags : int, optional
        Bartlett truncation lag.
    kind : {"Z(t_alpha)", "Z(alpha)"}
        Which statistic to report.

    Returns
    -------
    RichResult
        As :func:`phillips_perron_unit_root`.

    References
    ----------
    Phillips and Perron (1988), Biometrika 75:335-346; coded form from
    ``tseries::pp.test``.  See ``morie.fn.pptest`` for the full note.
    """
    if not trend:
        raise ValueError(
            "the tabulated critical values carried here are for the "
            "trend-included regression; trend=False is not available"
        )
    return phillips_perron_unit_root(y, lags=lags, kind=kind)


def cheatsheet():
    return "phillips_perron(y, trend=True, lags, kind): trend-argument spelling of the PP test."
