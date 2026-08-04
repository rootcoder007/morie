# morie.fn -- function file (rootcoder007/morie)
"""Local Moran's I per location (re-export)."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['lisamoran', 'local_morans_i']


def lisamoran(x, W, mlvar=True):
    """Local Moran's I per location (re-export).

    The shelf lists local Moran twice under different module names. One implementation, one delegating name.


    Formula: see localmoran

    Parameters
    ----------
    x : array-like
        Values at the n locations.
    W : array-like, shape (n, n)
        Spatial weights.
    mlvar : bool
        Divide m2 by n rather than n-1.

    Returns
    -------
    RichResult
        the payload of :func:`morie.fn.lismor.localmoran`.

    References
    ----------
    Anselin (1995), Local Indicators of Spatial Association -- LISA,
    Geographical Analysis 27(2):93-115, formula (12) p.99.  The
    article is paywalled; the formula and the divide-by-n variance
    convention were taken from spdep::localmoran, the reference
    implementation, which cites that equation explicitly.
    """
    from .lismor import localmoran as _lm
    return _lm(x, W, mlvar=mlvar)


local_morans_i = lisamoran


def cheatsheet():
    return "morloc: Local Moran's I per location (re-export)."
