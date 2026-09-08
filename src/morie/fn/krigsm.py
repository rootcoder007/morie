# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Ordinary kriging interpolation -- alias of :mod:`morie.fn.krig`.

Matheron (1963), "Principles of geostatistics", Economic Geology
58(8):1246-1266, doi:10.2113/gsecongeo.58.8.1246; Cressie (1993),
Statistics for Spatial Data, rev. ed., Wiley.

Ordinary kriging with a spherical variogram is already implemented in
:mod:`morie.fn.krig` under the same public name ``ordinary_kriging``;
this module re-exports it rather than carrying a second copy.
"""

from __future__ import annotations

from .krig import _spherical_variogram, ordinary_kriging  # noqa: F401

__all__ = ["ordinary_kriging"]


def cheatsheet():
    return "krigsm: Ordinary kriging interpolation -- alias of krig"
