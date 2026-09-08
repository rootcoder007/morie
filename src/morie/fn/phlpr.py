# morie.fn -- function file (rootcoder007/morie)
"""Phillips-Perron unit-root test (canonical entry point)."""

from __future__ import annotations

from .pptest import phillips_perron_unit_root

__all__ = ["phillips_perron"]


def phillips_perron(y, lags=None, kind="Z(t_alpha)"):
    """Phillips-Perron unit-root test.

    Thin re-export of :func:`morie.fn.pptest.phillips_perron_unit_root`,
    which carries the implementation, the Bartlett long-run variance and
    the Dickey-Fuller tables.  Three modules in this package spelled the
    same test three different ways; the arithmetic lives in one place so
    they cannot drift apart.

    Parameters
    ----------
    y : array-like
        Series in time order.
    lags : int, optional
        Bartlett truncation lag; the short-lag rule if omitted.
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
    return phillips_perron_unit_root(y, lags=lags, kind=kind)


def cheatsheet():
    return "phillips_perron(y, lags, kind): alias of pptest.phillips_perron_unit_root."


# compact alias per ledger/NAMING.md
phillipsperron = phillips_perron
