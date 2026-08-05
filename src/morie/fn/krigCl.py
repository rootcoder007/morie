# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Ordinary kriging -- alias of :mod:`morie.fn.krig`.

Matheron (1963), "Principles of geostatistics", Economic Geology
58(8):1246-1266, doi:10.2113/gsecongeo.58.8.1246.

Same estimator as :mod:`morie.fn.krigsm` and :mod:`morie.fn.krig`
(identical public name ``ordinary_kriging``, identical spherical
variogram system); re-exported rather than copied.
"""

from __future__ import annotations

from .krig import _spherical_variogram, ordinary_kriging  # noqa: F401

__all__ = ["ordinary_kriging"]


def cheatsheet():
    return "krigCl: Ordinary kriging -- alias of krig"
