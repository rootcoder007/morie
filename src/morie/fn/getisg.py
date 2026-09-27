"""Getis-Ord G global statistic.

The body that used to live here ran a one-sample Kolmogorov-Smirnov test of
``x`` against a fitted normal and never used ``W``. The statistic is the one
in :mod:`morie.fn.getsorg` (exact randomisation moments of Getis and Ord
1992, equal to ``spdep::globalG.test``), re-exported here.
"""

from .getsorg import getis_ord_g

__all__ = ["getis_ord_g"]


def cheatsheet():
    return "getis_ord_g(x, W) -> Getis-Ord global G with randomisation moments (= spdep::globalG.test)."


getisg = getis_ord_g
getisordg = getis_ord_g
